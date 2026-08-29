#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and run the native Cosmos3-Nano CP=2 workflow on two DGX Sparks.

This host-side orchestrator uses a locally built example image to build one
CP=2 TensorRT bundle on the primary Spark, copies the exact image and bundle to
a peer Spark, and launches one global rank on each host. Each scene has an
independent file rendezvous and runs only after the previous scene finishes.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
PRECISION = "bf16"
WIDTH = 1280
HEIGHT = 720
FRAME_COUNT = 189
FRAME_RATE = 24
STEPS = 35
GUIDANCE_SCALE = 6.0
FLOW_SHIFT = 10.0
CP_SIZE = 2
NCCL_ID_BYTES = 128
PHYSICS_NEGATIVE_PROMPT = (
    "blur, low detail, jitter, flicker, morphing, floating objects, interpenetrating "
    "objects, fused objects, teleportation, discontinuous motion, cuts, scene changes, "
    "text, captions, logos, watermarks"
)

DEFAULT_IMAGE = "trtmc-cosmos3-dual-spark:local"
DEFAULT_LOCAL_WORK_ROOT = Path.home() / ".cache" / "trtmc" / "cosmos3-physics"
DEFAULT_REMOTE_ROOT = "/var/tmp/cosmos3-physics-dual-spark"  # noqa: S108
DEFAULT_IB_HCA = "rocep1s0f0:1"
DEFAULT_NET_IFACE = "enp1s0f0np0"
DEFAULT_GID_INDEX = 3

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,47}$")
PEER_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REMOTE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9_./-]+$")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
NETWORK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

REMOTE_EXEC_HELPER = """\
import base64
import json
import subprocess
import sys

argv = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode("utf-8"))
raise SystemExit(subprocess.run(argv, check=False).returncode)
"""

REMOTE_ROOT_HELPER = """\
import os
import stat
import sys

root = sys.argv[1]
os.makedirs(os.path.dirname(root), exist_ok=True)
try:
    os.mkdir(root, 0o700)
except FileExistsError:
    pass

entry = os.lstat(root)
if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
    raise SystemExit("remote root must be a real directory")
if entry.st_uid != os.geteuid():
    raise SystemExit("remote root must be owned by the SSH user")

flags = os.O_RDONLY | os.O_DIRECTORY
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(root, flags)
try:
    opened = os.fstat(descriptor)
    if opened.st_uid != os.geteuid():
        raise SystemExit("remote root changed ownership during validation")
    os.fchmod(descriptor, 0o700)
    if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
        raise SystemExit("remote root permissions could not be restricted to 0700")
finally:
    os.close(descriptor)

for current_root, directories, filenames in os.walk(root, followlinks=False):
    for name in (*directories, *filenames):
        child = os.lstat(os.path.join(current_root, name))
        if stat.S_ISLNK(child.st_mode):
            raise SystemExit("remote root must not contain symbolic links")
        if child.st_uid != os.geteuid():
            raise SystemExit("remote root contents must be owned by the SSH user")
"""


class DualSparkError(RuntimeError):
    """A safe, operator-facing dual-Spark workflow error."""


@dataclass(frozen=True, slots=True)
class Scene:
    slug: str
    seed: int
    prompt: Mapping[str, Any]
    negative_prompt: str | Mapping[str, Any]


def _subject(
    description: str,
    *,
    appearance: str,
    relationship: str,
    location: str,
    size: str,
    orientation: str,
    pose: str,
    action: str,
    state_changes: str,
    count: int = 1,
    arms: int = 0,
    legs: int = 0,
) -> dict[str, object]:
    return {
        "description": description,
        "appearance_details": appearance,
        "relationship": relationship,
        "location": location,
        "relative_size": size,
        "orientation": orientation,
        "pose": pose,
        "action": action,
        "state_changes": state_changes,
        "clothing": "",
        "expression": "",
        "gender": "",
        "age": "",
        "skin_tone_and_texture": "",
        "facial_features": "",
        "number_of_subjects": count,
        "number_of_arms": arms,
        "number_of_legs": legs,
    }


def _showcase_prompt(
    *,
    subjects: list[dict[str, object]],
    background: str,
    light_conditions: str,
    light_direction: str,
    shadows: str,
    illumination: str,
    composition: str,
    colors: str,
    mood: str,
    camera_motion: str,
    framing: str,
    camera_angle: str,
    focus: str,
    lens: str,
    context: str,
    steps: tuple[str, str, str],
    temporal_caption: str,
) -> dict[str, object]:
    times = ("0:00-0:02", "0:02-0:05", "0:05-0:07.875")
    actions = [
        {"time": time_range, "description": description}
        for time_range, description in zip(times, steps, strict=True)
    ]
    segments = [
        {
            "segment_index": index,
            "time_range": time_range,
            "description": description,
            "key_changes": description,
            "camera": camera_motion,
        }
        for index, (time_range, description) in enumerate(
            zip(times, steps, strict=True)
        )
    ]
    return {
        "subjects": subjects,
        "background_setting": background,
        "lighting": {
            "conditions": light_conditions,
            "direction": light_direction,
            "shadows": shadows,
            "illumination_effect": illumination,
        },
        "aesthetics": {
            "composition": composition,
            "color_scheme": colors,
            "mood_atmosphere": mood,
            "patterns": "Natural, temporally stable surface detail without repetition.",
        },
        "cinematography": {
            "camera_motion": camera_motion,
            "framing": framing,
            "camera_angle": camera_angle,
            "depth_of_field": "Natural cinematic depth of field",
            "focus": focus,
            "lens_focal_length": lens,
        },
        "style_medium": "Photorealistic live-action video",
        "artistic_style": (
            "Physically plausible cinematic realism with stable identity, geometry, "
            "lighting, and textures in one continuous shot"
        ),
        "context": context,
        "actions": actions,
        "text_and_signage_elements": [],
        "segments": segments,
        "transitions": [],
        "temporal_caption": temporal_caption,
        "resolution": {"W": WIDTH, "H": HEIGHT},
        "aspect_ratio": "16,9",
        "duration": "7.875s",
        "fps": FRAME_RATE,
    }


SCENES = (
    Scene(
        slug="showcase-high-speed-racing",
        seed=21001,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "A low, aerodynamic red professional racing car navigating a winding "
                    "circuit at high speed.",
                    appearance=(
                        "Glossy red bodywork, intact aerodynamic surfaces, spinning black "
                        "tires, realistic suspension."
                    ),
                    relationship="Primary vehicle followed continuously through the turns",
                    location="Center of frame on the racing line",
                    size="Large within frame",
                    orientation="Moving away then carving through alternating bends",
                    pose="Tires firmly contacting the asphalt with body roll under load",
                    action=(
                        "Accelerating, braking, and steering through multiple winding turns"
                    ),
                    state_changes=(
                        "Suspension compresses and unloads naturally while the car advances "
                        "continuously"
                    ),
                )
            ],
            background=(
                "A closed professional road-racing circuit with curbs, barriers, distant "
                "grandstands, and clear safety runoff."
            ),
            light_conditions=(
                "Bright late-afternoon daylight with crisp atmospheric visibility"
            ),
            light_direction="Warm sun from camera-left with balanced sky fill",
            shadows="Stable moving car shadow and strong tire contact shadows",
            illumination="Specular highlights travel smoothly over the bodywork",
            composition=(
                "The car remains the clear focal point while successive bends reveal the "
                "track geometry."
            ),
            colors="Red car against gray asphalt, green verge, and blue sky",
            mood="Fast, controlled, exhilarating motorsport realism",
            camera_motion=(
                "One smooth low chase-camera tracking move with no cuts or teleportation"
            ),
            framing="Dynamic wide tracking shot that keeps the whole car visible",
            camera_angle="Low rear three-quarter angle near track height",
            focus="Sharp focus on the racing car with natural background motion blur",
            lens="35mm equivalent",
            context=(
                "A high-speed racing event in which a car navigates multiple winding turns."
            ),
            steps=(
                "The car accelerates into the first sweeping turn and loads the outside "
                "suspension.",
                "It changes direction through two connected bends while remaining planted "
                "on the racing line.",
                "It exits the final turn under acceleration and continues down the circuit.",
            ),
            temporal_caption=(
                "A single red racing car advances continuously through a sequence of winding "
                "turns, showing credible wheel rotation, steering, inertia, grip, and "
                "suspension response before accelerating away."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
    Scene(
        slug="showcase-mars-robots",
        seed=21003,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "Three rugged humanoid exploration robots operating together on the "
                    "surface of Mars.",
                    appearance=(
                        "White and graphite pressure-sealed shells, dust-coated boots, "
                        "compact sample canisters, blue status lights."
                    ),
                    relationship=(
                        "A coordinated three-robot survey team moving toward the same base"
                    ),
                    location="Across the center foreground and middle distance",
                    size="Medium to large within frame",
                    orientation=(
                        "Initially facing rock samples, then turning toward the base"
                    ),
                    pose="Balanced bipedal stances with feet planted in regolith",
                    action="Collecting geological samples and walking toward a Mars base",
                    state_changes=(
                        "Each robot secures a sample, stands, turns, and begins walking"
                    ),
                    count=3,
                    arms=6,
                    legs=6,
                )
            ],
            background=(
                "A broad rust-red Martian plain with rocks, shallow ridges, dusty tire "
                "tracks, and a pressurized Mars habitat in the near distance."
            ),
            light_conditions="Clear cold Martian daylight beneath a pale salmon sky",
            light_direction="Low sun from right with diffuse atmospheric fill",
            shadows="Long stable shadows attached to all three robots and rocks",
            illumination="Dusty warm bounce light with crisp metallic highlights",
            composition=(
                "All three robots and the destination habitat remain legible throughout "
                "the continuous shot."
            ),
            colors=(
                "Rust-red terrain, pale sky, white-and-graphite robots, blue status lights"
            ),
            mood="Purposeful scientific exploration and teamwork",
            camera_motion=(
                "Slow stabilized lateral tracking move that follows the team toward the base"
            ),
            framing="Wide shot with all three full robot bodies visible",
            camera_angle="Human eye-level landscape view",
            focus="Deep focus across robots, samples, footprints, and base",
            lens="32mm equivalent",
            context=(
                "Three humanoid robots on the surface of Mars collect samples and walk "
                "toward a Mars base."
            ),
            steps=(
                "The three robots crouch beside separate rocks and grasp geological samples.",
                "They place the samples into canisters, rise with balanced joint motion, "
                "and turn toward the habitat.",
                "The team walks together toward the Mars base, leaving sequential "
                "footprints in the dust.",
            ),
            temporal_caption=(
                "Three distinct humanoid robots collect one rock sample each, store the "
                "samples, stand without morphing, then walk in formation toward a visible "
                "Mars base while their feet remain grounded."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
    Scene(
        slug="showcase-delivery-robot",
        seed=21004,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "A compact four-wheeled autonomous delivery robot with a sealed cargo "
                    "compartment.",
                    appearance=(
                        "Matte white shell, black rubber tires, amber marker lights, rain "
                        "droplets on stable body panels."
                    ),
                    relationship="Primary vehicle traversing one shallow puddle",
                    location="Street level in the center of the alley",
                    size="Large within frame",
                    orientation="Rolling left-to-right past the camera",
                    pose="All four wheels grounded on wet pavement",
                    action="Rolling steadily through one shallow puddle",
                    state_changes=(
                        "Front wheels enter, displace water, and exit; rear wheels repeat "
                        "the displacement"
                    ),
                )
            ],
            background=(
                "A narrow rainy urban alley at night with wet asphalt, restrained cyan and "
                "magenta neon reflections, closed storefronts, and light rainfall."
            ),
            light_conditions="Night rain illuminated by practical neon and street lamps",
            light_direction=(
                "Colored side light from storefronts with soft overhead streetlight"
            ),
            shadows="Soft wet-ground contact shadow beneath the robot",
            illumination="Physically plausible reflections and specular rain highlights",
            composition=(
                "Low street-level view keeps the robot, its wheels, and the complete puddle "
                "interaction visible."
            ),
            colors=(
                "Neutral white robot, dark wet street, subtle cyan and magenta reflections"
            ),
            mood="Quiet, futuristic, believable last-mile delivery",
            camera_motion="Gentle street-level tracking pan with one continuous take",
            framing="Low medium-wide shot showing full robot and puddle",
            camera_angle="Camera approximately twenty centimeters above pavement",
            focus="Sharp robot and water-contact zone with soft alley depth",
            lens="40mm equivalent",
            context=(
                "A street-level camera shows a delivery robot rolling over one shallow "
                "puddle in a rainy neon alley, with realistic water ripples."
            ),
            steps=(
                "The delivery robot approaches one shallow puddle at constant speed as "
                "rain dimples the surface.",
                "Its front then rear wheels cross the puddle, pushing small waves and "
                "outward ripples without a large splash.",
                "The robot exits onto wet pavement while concentric ripples settle behind it.",
            ),
            temporal_caption=(
                "A four-wheeled delivery robot rolls continuously across exactly one shallow "
                "rain puddle; wheel contact creates small displaced waves and realistic "
                "ripples that remain and settle after it passes."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
    Scene(
        slug="showcase-apple-to-plate",
        seed=21006,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "A compact collaborative robot arm with a soft parallel gripper.",
                    appearance=(
                        "Smooth light-gray links, dark flexible joints, rubber fingertips, "
                        "small green status light."
                    ),
                    relationship="Picking up the apple and placing it on the plate",
                    location="Center above a tabletop",
                    size="Large within frame",
                    orientation="Moving from apple at left to plate at right",
                    pose="Stable base with gripper aligned around the apple",
                    action="Grasping, lifting, translating, and releasing one apple",
                    state_changes=(
                        "Empty open gripper closes on apple, carries it, opens above plate, "
                        "then retracts"
                    ),
                ),
                _subject(
                    "One glossy red apple and one clean white ceramic plate.",
                    appearance=(
                        "Single intact red apple with stem; round white plate with raised rim."
                    ),
                    relationship=(
                        "Apple is the transported object; plate is its destination"
                    ),
                    location="Apple at left foreground and plate at right foreground",
                    size="Medium within frame",
                    orientation="Both upright on the tabletop",
                    pose="Initially separate and stationary",
                    action="Apple moves only while held and ends centered on the plate",
                    state_changes=(
                        "Apple changes location from tabletop to plate without changing shape"
                    ),
                ),
            ],
            background=(
                "A clean demonstration table in a bright robotics laboratory with an "
                "uncluttered neutral backdrop."
            ),
            light_conditions="Soft bright studio daylight",
            light_direction="Large key from upper-left with frontal fill",
            shadows="Natural contact shadows under apple, plate, gripper, and robot",
            illumination="Gentle highlights preserve the apple and ceramic material",
            composition=(
                "The full pick-and-place path from apple to plate is visible at all times."
            ),
            colors="Red apple, white plate, light-gray robot, neutral background",
            mood="Clear, calm, reliable robotic dexterity",
            camera_motion="One fixed camera with a nearly imperceptible push-in",
            framing="Medium shot of complete robot workspace",
            camera_angle="Slightly high front view",
            focus="Sharp focus across gripper, apple, and plate",
            lens="50mm equivalent",
            context="The robot arm picks up the apple and places it on the plate.",
            steps=(
                "The open gripper approaches and closes gently around the single apple.",
                "The arm lifts the apple, translates smoothly above the table, and centers "
                "it over the plate.",
                "The apple is lowered into contact with the plate; the gripper opens and "
                "retracts.",
            ),
            temporal_caption=(
                "One unchanged apple is grasped, lifted clear of the table, carried in a "
                "smooth arc, placed at the center of one plate, released, and left "
                "stationary as the gripper retracts."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
    Scene(
        slug="showcase-humanoid-sprint",
        seed=21011,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "One athletic humanoid robot sprinting on a regulation running track.",
                    appearance=(
                        "Lightweight white-and-black shell, exposed precise joints, stable "
                        "head, red racing bib without text."
                    ),
                    relationship="Only runner on the track",
                    location="Center lane moving left-to-right",
                    size="Large within frame",
                    orientation="Leaning slightly forward in the direction of travel",
                    pose="Dynamic alternating sprint gait with coordinated arm swing",
                    action="Accelerating into a fast biomechanically plausible sprint",
                    state_changes=(
                        "Stride length and cadence increase while body identity and limb "
                        "count remain constant"
                    ),
                    arms=2,
                    legs=2,
                )
            ],
            background=(
                "An outdoor athletics stadium with a red synthetic track, white lane lines, "
                "green infield, and softly blurred empty stands."
            ),
            light_conditions="Bright clear morning daylight",
            light_direction="Sun from rear-left with open-sky fill",
            shadows="Stable running shadow and distinct foot contact shadows",
            illumination="Clean highlights define joint motion and track texture",
            composition=(
                "Full robot stays visible with lane lines emphasizing forward speed."
            ),
            colors="Red track, green infield, white-and-black robot, blue sky",
            mood="Powerful, focused, technically impressive athletic motion",
            camera_motion=(
                "Smooth side-tracking camera matching the robot's speed in one continuous shot"
            ),
            framing="Full-body medium-wide profile shot",
            camera_angle="Waist-height side view",
            focus="Sharp robot with controlled horizontal background motion blur",
            lens="45mm equivalent",
            context="A humanoid robot sprinting on an athletic track.",
            steps=(
                "The humanoid robot drives out of acceleration with a grounded right-foot "
                "push and coordinated opposite arm swing.",
                "It reaches a fast alternating sprint cycle with clear airborne and "
                "single-foot support phases.",
                "The robot sustains speed down its lane with stable posture, rhythmic "
                "strides, and no foot sliding.",
            ),
            temporal_caption=(
                "One unchanged two-armed, two-legged humanoid robot accelerates and sprints "
                "down a marked lane using a coherent alternating gait, correct foot "
                "contacts, natural joint timing, and synchronized arm swing."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
    Scene(
        slug="showcase-cake-cutting",
        seed=21012,
        prompt=_showcase_prompt(
            subjects=[
                _subject(
                    "Two robotic hands viewed from their own head-mounted camera, one "
                    "stabilizing a cake and one holding a cake knife.",
                    appearance=(
                        "Matching white bimanual grippers, black joint covers, clean "
                        "stainless-steel knife."
                    ),
                    relationship="Coordinated bimanual system cutting one cake",
                    location="Lower left and lower right foreground in ego view",
                    size="Large foreground elements",
                    orientation="Both hands directed toward the cake",
                    pose=(
                        "One hand gently stabilizes the cake board while the other aligns "
                        "the knife"
                    ),
                    action="Making one smooth precise cut through the cake",
                    state_changes=(
                        "Knife descends and advances; cake separates cleanly; knife withdraws"
                    ),
                    count=2,
                    arms=2,
                ),
                _subject(
                    "One small round layered cake on a ceramic serving plate.",
                    appearance=(
                        "Stable vanilla frosting, subtle berry decoration, visible layered "
                        "interior after cutting."
                    ),
                    relationship="Object being cut by the two robotic hands",
                    location="Center of the work surface",
                    size="Medium to large within frame",
                    orientation="Upright and centered on the plate",
                    pose="Stationary on a nonslip mat",
                    action="Separating along one clean radial cut",
                    state_changes=(
                        "Changes from intact cake to one visible clean cut without deformation"
                    ),
                ),
            ],
            background=(
                "A quiet modern home kitchen worktop with uncluttered utensils and soft "
                "neutral cabinetry."
            ),
            light_conditions="Soft indoor daylight with warm overhead fill",
            light_direction="Window light from left and diffused ceiling light",
            shadows="Gentle stable hand, knife, cake, and plate contact shadows",
            illumination="Natural frosting texture and restrained metal highlights",
            composition=(
                "Ego-view centers the knife path and keeps both robotic hands visible."
            ),
            colors=(
                "Warm neutral kitchen, white frosting, muted berries, white robotic hands"
            ),
            mood="Calm, precise, careful domestic assistance",
            camera_motion=(
                "Stable first-person head camera with only subtle natural robot-body "
                "motion and no cuts"
            ),
            framing="Ego-view medium close-up of both hands and cake",
            camera_angle="Slight downward first-person angle",
            focus="Sharp knife edge, cake contact, and both grippers",
            lens="35mm equivalent",
            context=(
                "Ego-view footage from a bimanual robot's camera as it cuts a cake, with "
                "smooth, precise motion and soft indoor light."
            ),
            steps=(
                "The stabilizing hand rests beside the cake while the knife hand aligns "
                "the blade over one radial cut.",
                "The blade descends and moves smoothly through frosting and layers as the "
                "other hand prevents the plate from sliding.",
                "The cut completes cleanly; the knife lifts and both hands pause with the "
                "cake stable on the plate.",
            ),
            temporal_caption=(
                "From one continuous bimanual robot ego-view, one hand stabilizes the plate "
                "while the other makes exactly one smooth knife cut through an unchanged "
                "cake and then withdraws."
            ),
        ),
        negative_prompt=PHYSICS_NEGATIVE_PROMPT,
    ),
)
SCENE_BY_SLUG = {scene.slug: scene for scene in SCENES}
SCENE_CHOICES = ("all", *SCENE_BY_SLUG)


@dataclass(frozen=True, slots=True)
class Settings:
    action: str
    scene: str
    peer_host: str
    peer_user: str
    ssh_key: Path | None
    known_hosts: Path | None
    work_root: Path
    remote_root: str
    image: str
    run_id: str
    ib_hca: str
    net_iface: str
    gid_index: int
    runtime_timeout: int
    build_timeout: int
    dry_run: bool

    @property
    def peer_target(self) -> str:
        return f"{self.peer_user}@{self.peer_host}"


@dataclass(slots=True)
class MemoryTracker:
    """Byte-exact cgroup-v2 peak counters for one launched rank."""

    container: str
    remote: bool
    cgroup_path: str
    peak_bytes: int = 0
    swap_peak_bytes: int = 0
    samples: int = 0
    records: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    initial_events: str = ""
    latest_events: str = ""


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("physics-%Y%m%d-%H%M%S")


def _profile() -> dict[str, int | float]:
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAME_COUNT,
        "fps": FRAME_RATE,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": FLOW_SHIFT,
        "context_parallel_size": CP_SIZE,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Prepare and run {len(SCENES)} native 720p Cosmos3-Nano CP=2 physics scenes "
            "across two one-GPU DGX Sparks."
        )
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("all", "prepare", "run"),
        default="all",
        help="workflow action (default: all)",
    )
    parser.add_argument(
        "--scene",
        choices=SCENE_CHOICES,
        default=os.environ.get("COSMOS3_SCENE", "all"),
        help="generate all fixed scenes or one scene slug (default: all)",
    )
    parser.add_argument("--peer-host", default=os.environ.get("COSMOS3_PEER_HOST", ""))
    parser.add_argument(
        "--peer-user",
        default=os.environ.get("COSMOS3_PEER_USER", getpass.getuser()),
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=(
            Path(os.environ["COSMOS3_SSH_KEY"])
            if os.environ.get("COSMOS3_SSH_KEY")
            else None
        ),
    )
    parser.add_argument(
        "--known-hosts",
        type=Path,
        default=(
            Path(os.environ["COSMOS3_KNOWN_HOSTS"])
            if os.environ.get("COSMOS3_KNOWN_HOSTS")
            else None
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(
            os.environ.get(
                "COSMOS3_DUAL_SPARK_ROOT",
                str(DEFAULT_LOCAL_WORK_ROOT),
            )
        ),
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("COSMOS3_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
    )
    parser.add_argument("--image", default=os.environ.get("COSMOS3_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--run-id", default=os.environ.get("COSMOS3_RUN_ID", _default_run_id()))
    parser.add_argument("--ib-hca", default=os.environ.get("COSMOS3_IB_HCA", DEFAULT_IB_HCA))
    parser.add_argument(
        "--net-iface",
        default=os.environ.get("COSMOS3_NET_IFACE", DEFAULT_NET_IFACE),
    )
    parser.add_argument(
        "--gid-index",
        type=int,
        default=int(os.environ.get("COSMOS3_GID_INDEX", DEFAULT_GID_INDEX)),
    )
    parser.add_argument("--runtime-timeout", type=int, default=2 * 60 * 60)
    parser.add_argument("--build-timeout", type=int, default=12 * 60 * 60)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a mutation-free JSON execution plan",
    )
    return parser


def _settings(arguments: argparse.Namespace) -> Settings:
    settings = Settings(
        action=arguments.action,
        scene=arguments.scene,
        peer_host=arguments.peer_host,
        peer_user=arguments.peer_user,
        ssh_key=arguments.ssh_key.resolve() if arguments.ssh_key else None,
        known_hosts=arguments.known_hosts.resolve() if arguments.known_hosts else None,
        work_root=arguments.work_root.resolve(),
        remote_root=arguments.remote_root.rstrip("/"),
        image=arguments.image,
        run_id=arguments.run_id,
        ib_hca=arguments.ib_hca.removeprefix("="),
        net_iface=arguments.net_iface.removeprefix("="),
        gid_index=arguments.gid_index,
        runtime_timeout=arguments.runtime_timeout,
        build_timeout=arguments.build_timeout,
        dry_run=arguments.dry_run,
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    if not RUN_ID_PATTERN.fullmatch(settings.run_id):
        raise DualSparkError("--run-id must match [a-z0-9][a-z0-9_.-]{0,47}")
    if not PEER_COMPONENT_PATTERN.fullmatch(settings.peer_host):
        raise DualSparkError("--peer-host is required and must be a DNS name or IPv4 address")
    if not PEER_COMPONENT_PATTERN.fullmatch(settings.peer_user):
        raise DualSparkError("--peer-user contains unsupported characters")
    if not IMAGE_PATTERN.fullmatch(settings.image):
        raise DualSparkError("--image contains unsupported characters")
    if str(settings.work_root) == "/" or "," in str(settings.work_root):
        raise DualSparkError("--work-root must be scoped and must not contain a comma")
    if not REMOTE_PATH_PATTERN.fullmatch(settings.remote_root):
        raise DualSparkError("--remote-root must be an absolute path without spaces")
    if settings.remote_root == "/" or ".." in Path(settings.remote_root).parts:
        raise DualSparkError("--remote-root must be a scoped directory")
    if not NETWORK_NAME_PATTERN.fullmatch(settings.ib_hca):
        raise DualSparkError("--ib-hca contains unsupported characters")
    if not NETWORK_NAME_PATTERN.fullmatch(settings.net_iface):
        raise DualSparkError("--net-iface contains unsupported characters")
    try:
        _hca_name, port_text = settings.ib_hca.rsplit(":", 1)
        port = int(port_text)
    except (TypeError, ValueError) as exc:
        raise DualSparkError(
            "--ib-hca must use HCA:PORT form, for example rocep1s0f0:1"
        ) from exc
    if port < 1:
        raise DualSparkError("--ib-hca port must be positive")
    if settings.gid_index < 0:
        raise DualSparkError("--gid-index must be non-negative")
    if settings.runtime_timeout <= 0 or settings.build_timeout <= 0:
        raise DualSparkError("timeouts must be positive")

    local_names = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
    if settings.peer_host in local_names:
        raise DualSparkError("--peer-host must identify the second DGX Spark")


def _selected_scenes(settings: Settings) -> tuple[Scene, ...]:
    if settings.scene == "all":
        return SCENES
    return (SCENE_BY_SLUG[settings.scene],)


def _ssh_argv(settings: Settings) -> list[str]:
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if settings.ssh_key is not None:
        argv.extend(["-i", str(settings.ssh_key), "-o", "IdentitiesOnly=yes"])
    if settings.known_hosts is not None:
        argv.extend(["-o", f"UserKnownHostsFile={settings.known_hosts}"])
    argv.append(settings.peer_target)
    return argv


def _scp_argv(settings: Settings) -> list[str]:
    argv = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    if settings.ssh_key is not None:
        argv.extend(["-i", str(settings.ssh_key), "-o", "IdentitiesOnly=yes"])
    if settings.known_hosts is not None:
        argv.extend(["-o", f"UserKnownHostsFile={settings.known_hosts}"])
    return argv


def _layout(settings: Settings) -> dict[str, Any]:
    models = settings.work_root / "models"
    run_root = settings.work_root / "runs" / settings.run_id
    return {
        "hf_cache": settings.work_root / "hf-cache",
        "models": models,
        "bundle": models / "cosmos3-nano-cp2-1280x720.trtfb",
        "remote_bundle": f"{settings.remote_root}/models/cosmos3-nano-cp2-1280x720.trtfb",
        "preparation": settings.work_root / "preparation.json",
        "run_root": run_root,
        "remote_run_root": f"{settings.remote_root}/runs/{settings.run_id}",
        "run_json": run_root / "run.json",
    }


def _scene_layout(settings: Settings, scene: Scene) -> dict[str, Any]:
    layout = _layout(settings)
    scene_root = Path(layout["run_root"]) / scene.slug
    remote_scene_root = f"{layout['remote_run_root']}/{scene.slug}"
    rendezvous_name = f"{scene.slug}-seed{scene.seed}.nccl"
    return {
        "scene_root": scene_root,
        "rank0": scene_root / "rank0",
        "rank1_capture": scene_root / "rank1",
        "remote_rank1": f"{remote_scene_root}/rank1",
        "rendezvous": scene_root / "rendezvous" / rendezvous_name,
        "remote_rendezvous": f"{remote_scene_root}/rendezvous/{rendezvous_name}",
        "mp4": Path(layout["run_root"]) / f"{scene.slug}-720p.mp4",
        "rank0_container": f"cosmos3-{settings.run_id}-{scene.slug}-rank0",
        "rank1_container": f"cosmos3-{settings.run_id}-{scene.slug}-rank1",
    }


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _prompt_text(scene: Scene) -> str:
    return _compact_json(scene.prompt)


def _negative_prompt_text(scene: Scene) -> str:
    if isinstance(scene.negative_prompt, str):
        return scene.negative_prompt
    return _compact_json(scene.negative_prompt)


def _prompt_record(scene: Scene) -> dict[str, Any]:
    return {
        "positive": dict(scene.prompt),
        "negative": (
            scene.negative_prompt
            if isinstance(scene.negative_prompt, str)
            else dict(scene.negative_prompt)
        ),
    }


def _generation_argv(scene: Scene) -> list[str]:
    return [
        "/opt/trtmc/bin/trtmc",
        "generate-video",
        "/models/cosmos3.trtfb",
        "--prompt",
        _prompt_text(scene),
        "--negative-prompt",
        _negative_prompt_text(scene),
        "--output",
        "/outputs/frames",
        "--seed",
        str(scene.seed),
        "--num-steps",
        str(STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--height",
        str(HEIGHT),
        "--width",
        str(WIDTH),
    ]


def _rank_environment(settings: Settings, scene: Scene, rank: int) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "WORLD_SIZE": str(CP_SIZE),
        "RANK": str(rank),
        "LOCAL_RANK": "0",
        "TRTMC_NCCL_RENDEZVOUS": f"/rendezvous/{scene.slug}-seed{scene.seed}.nccl",
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": f"={settings.ib_hca}",
        "NCCL_IB_GID_INDEX": str(settings.gid_index),
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_SOCKET_IFNAME": f"={settings.net_iface}",
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _rank_docker_argv(
    settings: Settings,
    scene: Scene,
    rank: int,
    *,
    uverbs_device: str,
) -> list[str]:
    layout = _layout(settings)
    scene_paths = _scene_layout(settings, scene)
    if rank == 0:
        bundle = str(layout["bundle"])
        output_dir = str(scene_paths["rank0"])
        rendezvous_dir = str(Path(scene_paths["rendezvous"]).parent)
        container = str(scene_paths["rank0_container"])
    elif rank == 1:
        bundle = str(layout["remote_bundle"])
        output_dir = str(scene_paths["remote_rank1"])
        rendezvous_dir = str(Path(scene_paths["remote_rendezvous"]).parent)
        container = str(scene_paths["rank1_container"])
    else:
        raise ValueError("rank must be 0 or 1")

    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--no-healthcheck",
        "--label",
        "trtmc.example=cosmos3-physics-dual-spark",
        "--label",
        f"trtmc.run_id={settings.run_id}",
        "--label",
        f"trtmc.scene={scene.slug}",
        "--label",
        f"trtmc.rank={rank}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--gpus",
        "all",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add",
        "IPC_LOCK",
        "--device",
        "/dev/infiniband/rdma_cm",
        "--device",
        uverbs_device,
    ]
    for name, value in _rank_environment(settings, scene, rank).items():
        argv.extend(["-e", f"{name}={value}"])
    argv.extend(
        [
            "--mount",
            f"type=bind,src={bundle},dst=/models/cosmos3.trtfb,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/outputs",
            "--mount",
            f"type=bind,src={rendezvous_dir},dst=/rendezvous",
            "--entrypoint",
            "/opt/trtmc/bin/trtmc",
            settings.image,
            *_generation_argv(scene)[1:],
        ]
    )
    return argv


def _bundle_build_argv(settings: Settings) -> list[str]:
    layout = _layout(settings)
    pending_name = f"{Path(layout['bundle']).name}.pending"
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add",
        "IPC_LOCK",
        "--mount",
        f"type=bind,src={layout['hf_cache']},dst=/root/.cache/huggingface",
        "--mount",
        f"type=bind,src={layout['models']},dst=/models",
        "--entrypoint",
        "/opt/trtmc/bin/trtmc",
        settings.image,
        "build",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--precision",
        PRECISION,
        "--context-parallel-size",
        str(CP_SIZE),
        "--video-height",
        str(HEIGHT),
        "--video-width",
        str(WIDTH),
        "--video-num-frames",
        str(FRAME_COUNT),
        "--num-inference-steps",
        str(STEPS),
        "-o",
        f"/models/{pending_name}",
    ]


def execution_plan(settings: Settings) -> dict[str, Any]:
    layout = _layout(settings)
    hca_name = settings.ib_hca.rsplit(":", 1)[0]
    planned_uverbs = {
        "primary": "/dev/infiniband/<resolved-on-primary>",
        "peer": "/dev/infiniband/<resolved-on-peer>",
    }
    scene_plans = []
    for scene in _selected_scenes(settings):
        paths = _scene_layout(settings, scene)
        scene_plans.append(
            {
                "slug": scene.slug,
                "seed": scene.seed,
                "prompt": _prompt_record(scene),
                "rendezvous": {
                    "bytes": NCCL_ID_BYTES,
                    "primary": str(paths["rendezvous"]),
                    "peer": str(paths["remote_rendezvous"]),
                },
                "ranks": [
                    {
                        "host": "primary",
                        "rank": 0,
                        "docker_argv": _rank_docker_argv(
                            settings, scene, 0, uverbs_device=planned_uverbs["primary"]
                        ),
                    },
                    {
                        "host": settings.peer_target,
                        "rank": 1,
                        "docker_argv": _rank_docker_argv(
                            settings, scene, 1, uverbs_device=planned_uverbs["peer"]
                        ),
                    },
                ],
                "mp4": str(paths["mp4"]),
                "encoding": {
                    "codec": "libx264",
                    "crf": 18,
                    "preset": "slow",
                    "pixel_format": "yuv420p",
                    "faststart": True,
                },
            }
        )

    return {
        "schema_version": 1,
        "action": settings.action,
        "scene_selection": settings.scene,
        "selected_scene_slugs": [scene.slug for scene in _selected_scenes(settings)],
        "dry_run": settings.dry_run,
        "run_id": settings.run_id,
        "peer": settings.peer_target,
        "host_validation": {
            "authoritative_identity": "NVIDIA GPU UUID",
            "require_distinct_gpu_uuids": True,
            "require_matching_gpu_name_compute_capability_driver": True,
            "machine_id_sources": ["/etc/machine-id", "/var/lib/dbus/machine-id"],
            "machine_ids_are_advisory_only": True,
        },
        "ssh": {
            "argv": _ssh_argv(settings),
            "scp_argv": _scp_argv(settings),
            "batch_mode": True,
            "strict_host_key_checking": True,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": PRECISION},
        "profile": _profile(),
        "roce": {
            "hca": settings.ib_hca,
            "gid_index": settings.gid_index,
            "network_interface": settings.net_iface,
            "address_family": "AF_INET",
            "uverbs_resolution": {
                "sysfs_directory": (
                    f"/sys/class/infiniband/{hca_name}/device/infiniband_verbs"
                ),
                "primary_device": planned_uverbs["primary"],
                "peer_device": planned_uverbs["peer"],
                "resolved_at_runtime": True,
            },
        },
        "prepare": {
            "image": settings.image,
            "image_source": "prebuilt locally from this example's Dockerfile",
            "image_sync": "docker image save on primary -> strict SSH -> docker image load",
            "bundle_build_argv": _bundle_build_argv(settings),
            "bundle": str(layout["bundle"]),
            "peer_bundle": str(layout["remote_bundle"]),
            "require_matching_image_id": True,
            "require_matching_bundle_sha256": True,
        },
        "run": {
            "order": [scene.slug for scene in _selected_scenes(settings)],
            "scenes": scene_plans,
            "run_json": str(layout["run_json"]),
            "telemetry": {
                "performance": "parsed from the rank-0 [cosmos3.perf] record",
                "memory": "sampled from each rank's cgroup-v2 byte counters",
            },
        },
    }


def _log(message: str) -> None:
    print(f"[cosmos3-dual-spark] {message}", file=sys.stderr, flush=True)


def _run(
    argv: Sequence[str],
    *,
    check: bool = True,
    timeout: int | float | None = None,
    input_text: str | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    _log(f"$ {shlex.join(str(value) for value in argv)}")
    try:
        completed = subprocess.run(
            [str(value) for value in argv],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            stdin=None if input_text is not None else subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DualSparkError(f"command timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise DualSparkError(f"cannot start required command: {argv[0]}") from exc
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        suffix = f": {detail}" if detail else ""
        raise DualSparkError(
            f"command failed with exit code {completed.returncode}{suffix}"
        )
    return completed


def _remote_run(
    settings: Settings,
    argv: Sequence[str],
    *,
    check: bool = True,
    timeout: int | float | None = None,
) -> subprocess.CompletedProcess[str]:
    payload = base64.urlsafe_b64encode(
        json.dumps([str(value) for value in argv]).encode("utf-8")
    ).decode("ascii")
    remote_command = f"python3 - {shlex.quote(payload)}"
    return _run(
        [*_ssh_argv(settings), remote_command],
        check=check,
        timeout=timeout,
        input_text=REMOTE_EXEC_HELPER,
    )


def _ensure_private_remote_root(settings: Settings) -> None:
    _remote_run(
        settings,
        ["python3", "-c", REMOTE_ROOT_HELPER, settings.remote_root],
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_sha256(settings: Settings, path: str) -> str:
    output = _remote_run(settings, ["sha256sum", "--", path]).stdout.strip()
    digest = output.split(maxsplit=1)[0] if output else ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DualSparkError(f"peer returned an invalid SHA-256 for {path}")
    return digest


def _image_id(settings: Settings, *, remote: bool) -> str:
    argv = ["docker", "image", "inspect", "--format", "{{.Id}}", settings.image]
    completed = (
        _remote_run(settings, argv, check=False)
        if remote
        else _run(argv, check=False)
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _gpu_facts(settings: Settings, *, remote: bool) -> dict[str, str]:
    argv = [
        "nvidia-smi",
        "--query-gpu=uuid,name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = _remote_run(settings, argv) if remote else _run(argv)
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    location = "peer" if remote else "primary"
    if len(rows) != 1:
        raise DualSparkError(
            f"{location} DGX Spark must expose exactly one GPU; found {len(rows)}"
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 4:
        raise DualSparkError(f"could not parse {location} nvidia-smi GPU identity")
    facts = {
        "uuid": parts[0],
        "name": parts[1],
        "compute_capability": parts[2],
        "driver": parts[3],
    }
    if not re.fullmatch(r"GPU-[0-9A-Fa-f-]+", facts["uuid"]):
        raise DualSparkError(f"could not parse {location} nvidia-smi GPU UUID")
    if "GB10" not in facts["name"].upper() or facts["compute_capability"] != "12.1":
        raise DualSparkError(
            f"{location} is not the required one-GPU GB10 DGX Spark (reported "
            f"{facts['name']!r}, compute capability {facts['compute_capability']!r})"
        )
    return facts


def _read_host_file(settings: Settings, path: str, *, remote: bool) -> str:
    argv = ["cat", "--", path]
    completed = _remote_run(settings, argv) if remote else _run(argv)
    return completed.stdout.strip()


def _machine_id(settings: Settings, *, remote: bool) -> str | None:
    """Return an advisory OS image identifier when one is available.

    DGX Sparks may legitimately share a machine ID after an OS image is cloned,
    so this value must never be used to decide whether the hosts are distinct.
    """

    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        argv = ["cat", "--", path]
        completed = (
            _remote_run(settings, argv, check=False)
            if remote
            else _run(argv, check=False)
        )
        candidate = completed.stdout.strip().lower()
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{32}", candidate):
            return candidate
    return None


def _recorded_gpu_facts(facts: Mapping[str, str]) -> dict[str, str]:
    """Return provenance-safe GPU facts without publishing the raw UUID."""

    return {
        "name": facts["name"],
        "compute_capability": facts["compute_capability"],
        "driver": facts["driver"],
        "uuid_sha256": hashlib.sha256(facts["uuid"].encode("ascii")).hexdigest(),
    }


def _validate_host_identity(
    local_gpu: Mapping[str, str],
    peer_gpu: Mapping[str, str],
    local_machine_id: str | None,
    peer_machine_id: str | None,
) -> dict[str, Any]:
    """Validate two matching Spark stacks using GPU UUIDs as host identity."""

    for fact_name in ("name", "compute_capability", "driver"):
        if local_gpu[fact_name] != peer_gpu[fact_name]:
            raise DualSparkError(
                "the two Sparks must use matching GPU/driver stacks "
                f"({fact_name} differs)"
            )
    if local_gpu["uuid"] == peer_gpu["uuid"]:
        raise DualSparkError(
            "--peer-host resolves to the primary Spark; two distinct NVIDIA GPU UUIDs "
            "are required"
        )

    def optional_hash(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("ascii")).hexdigest() if value else None

    return {
        "authoritative_source": "nvidia_gpu_uuid",
        "primary_gpu_uuid_sha256": hashlib.sha256(
            local_gpu["uuid"].encode("ascii")
        ).hexdigest(),
        "peer_gpu_uuid_sha256": hashlib.sha256(
            peer_gpu["uuid"].encode("ascii")
        ).hexdigest(),
        "machine_ids_authoritative": False,
        "primary_machine_id_sha256": optional_hash(local_machine_id),
        "peer_machine_id_sha256": optional_hash(peer_machine_id),
        "machine_ids_match": (
            local_machine_id == peer_machine_id
            if local_machine_id is not None and peer_machine_id is not None
            else None
        ),
    }


def _resolve_uverbs_device(
    settings: Settings,
    hca_name: str,
    *,
    remote: bool,
) -> str:
    location = "peer" if remote else "primary"
    verbs_directory = f"/sys/class/infiniband/{hca_name}/device/infiniband_verbs"
    argv = ["ls", "-1", "--", verbs_directory]
    completed = _remote_run(settings, argv) if remote else _run(argv)
    names = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if re.fullmatch(r"uverbs[0-9]+", line.strip())
    )
    if len(names) != 1:
        raise DualSparkError(
            f"{location} HCA {hca_name!r} must resolve to exactly one uverbs device; "
            f"found {names}"
        )
    device = f"/dev/infiniband/{names[0]}"
    check_argv = ["test", "-c", device]
    checked = (
        _remote_run(settings, check_argv, check=False)
        if remote
        else _run(check_argv, check=False)
    )
    if checked.returncode != 0:
        raise DualSparkError(f"{location} resolved uverbs device is unavailable: {device}")
    return device


def _preflight(settings: Settings) -> dict[str, Any]:
    for executable in ("docker", "ssh", "scp", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise DualSparkError(f"required executable is unavailable: {executable}")
    for label, path in (
        ("SSH identity", settings.ssh_key),
        ("known-hosts", settings.known_hosts),
    ):
        if path is not None and (not path.is_file() or not os.access(path, os.R_OK)):
            raise DualSparkError(f"{label} file is not readable: {path}")

    _run(["docker", "info"], timeout=60)
    _remote_run(settings, ["docker", "info"], timeout=60)
    _remote_run(settings, ["python3", "--version"], timeout=30)

    local_gpu = _gpu_facts(settings, remote=False)
    peer_gpu = _gpu_facts(settings, remote=True)
    local_machine_id = _machine_id(settings, remote=False)
    peer_machine_id = _machine_id(settings, remote=True)
    host_identity = _validate_host_identity(
        local_gpu, peer_gpu, local_machine_id, peer_machine_id
    )

    hca_name, port_text = settings.ib_hca.rsplit(":", 1)
    port = int(port_text)
    resource_checks = (
        ("-d", f"/sys/class/net/{settings.net_iface}"),
        ("-d", f"/sys/class/infiniband/{hca_name}"),
        ("-c", "/dev/infiniband/rdma_cm"),
        (
            "-f",
            f"/sys/class/infiniband/{hca_name}/ports/{port}/gids/{settings.gid_index}",
        ),
    )
    for predicate, path in resource_checks:
        local = _run(["test", predicate, path], check=False)
        peer = _remote_run(settings, ["test", predicate, path], check=False)
        if local.returncode != 0 or peer.returncode != 0:
            raise DualSparkError(f"required RoCE resource is missing on one or both Sparks: {path}")

    interface_state_path = f"/sys/class/net/{settings.net_iface}/operstate"
    hca_state_path = f"/sys/class/infiniband/{hca_name}/ports/{port}/state"
    gid_path = f"/sys/class/infiniband/{hca_name}/ports/{port}/gids/{settings.gid_index}"
    roce: dict[str, dict[str, str]] = {}
    for label, remote in (("primary", False), ("peer", True)):
        uverbs_device = _resolve_uverbs_device(settings, hca_name, remote=remote)
        interface_state = _read_host_file(
            settings, interface_state_path, remote=remote
        ).lower()
        hca_state = _read_host_file(settings, hca_state_path, remote=remote).upper()
        gid = _read_host_file(settings, gid_path, remote=remote)
        if interface_state != "up":
            raise DualSparkError(f"{label} RoCE network interface is not up")
        if "ACTIVE" not in hca_state:
            raise DualSparkError(f"{label} RoCE HCA port is not active")
        if not gid or gid in {"::", "0:0:0:0:0:0:0:0"}:
            raise DualSparkError(f"{label} RoCE GID index does not contain a usable address")
        roce[label] = {
            "interface_state": interface_state,
            "hca_state": hca_state,
            "gid": gid,
            "uverbs_device": uverbs_device,
        }

    return {
        "primary_gpu": _recorded_gpu_facts(local_gpu),
        "peer_gpu": _recorded_gpu_facts(peer_gpu),
        "host_identity": host_identity,
        "roce_hosts": roce,
    }


def _sync_image(settings: Settings, expected_id: str) -> None:
    if _image_id(settings, remote=True) == expected_id:
        _log("peer already has the exact image ID")
        return
    _log("copying the exact image to the peer Spark")
    with tempfile.TemporaryFile() as save_stderr_file:
        try:
            save = subprocess.Popen(
                ["docker", "image", "save", settings.image],
                stdout=subprocess.PIPE,
                stderr=save_stderr_file,
                shell=False,
            )
        except OSError as exc:
            raise DualSparkError("cannot start docker image save") from exc
        assert save.stdout is not None
        try:
            load = subprocess.Popen(
                [*_ssh_argv(settings), "docker image load"],
                stdin=save.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError as exc:
            save.kill()
            save.wait()
            raise DualSparkError("cannot start peer docker image load") from exc
        save.stdout.close()
        try:
            load_stdout, load_stderr = load.communicate(timeout=settings.build_timeout)
            save_returncode = save.wait(timeout=30)
            save_stderr_file.seek(0)
            save_stderr = save_stderr_file.read()
        except subprocess.TimeoutExpired as exc:
            load.kill()
            save.kill()
            load.wait()
            save.wait()
            raise DualSparkError("timed out while copying the image to the peer") from exc
    if save_returncode != 0 or load.returncode != 0:
        detail = (save_stderr + load_stdout + load_stderr).decode("utf-8", "replace")
        raise DualSparkError(f"image transfer failed: {detail[-2000:]}")
    if _image_id(settings, remote=True) != expected_id:
        raise DualSparkError("peer image ID does not match the primary after transfer")


def _bundle_spec(image_id: str) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "precision": PRECISION,
        **_profile(),
    }


def _prepare_bundle(settings: Settings, image_id: str) -> Path:
    layout = _layout(settings)
    bundle = Path(layout["bundle"])
    marker = bundle.with_suffix(f"{bundle.suffix}.build-spec.json")
    expected = _bundle_spec(image_id)
    if bundle.is_file() and bundle.stat().st_size > 0 and marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected:
                _log("reusing the exact native 720p CP=2 TensorRT bundle")
                return bundle
        except (OSError, json.JSONDecodeError):
            pass

    pending = bundle.with_name(f"{bundle.name}.pending")
    if pending.exists():
        pending.unlink()
    _log("building the native 720p CP=2 TensorRT bundle; this can take hours")
    _run(_bundle_build_argv(settings), timeout=settings.build_timeout, capture=False)
    if not pending.is_file() or pending.stat().st_size == 0:
        raise DualSparkError("bundle build completed without a nonempty CP=2 bundle")
    os.replace(pending, bundle)
    _write_json(marker, expected)
    return bundle


def _sync_bundle(settings: Settings, bundle: Path) -> str:
    _ensure_private_remote_root(settings)
    layout = _layout(settings)
    remote_bundle = str(layout["remote_bundle"])
    remote_parent = str(Path(remote_bundle).parent)
    _remote_run(settings, ["install", "-d", "-m", "0700", remote_parent])
    digest = _sha256(bundle)
    existing = _remote_run(settings, ["test", "-f", remote_bundle], check=False)
    if existing.returncode == 0 and _remote_sha256(settings, remote_bundle) == digest:
        _log("peer already has the exact CP=2 bundle SHA-256")
        return digest

    incoming = f"{remote_bundle}.incoming.{settings.run_id}"
    _remote_run(settings, ["rm", "-f", "--", incoming], check=False)
    transferred = False
    try:
        _run(
            [*_scp_argv(settings), str(bundle), f"{settings.peer_target}:{incoming}"],
            timeout=settings.build_timeout,
            capture=False,
        )
        if _remote_sha256(settings, incoming) != digest:
            raise DualSparkError("peer bundle checksum does not match the primary")
        _remote_run(settings, ["mv", "-T", "--", incoming, remote_bundle])
        transferred = True
    finally:
        if not transferred:
            _remote_run(settings, ["rm", "-f", "--", incoming], check=False)
    return digest


def prepare(settings: Settings) -> dict[str, Any]:
    layout = _layout(settings)
    Path(layout["hf_cache"]).mkdir(parents=True, exist_ok=True)
    Path(layout["models"]).mkdir(parents=True, exist_ok=True)
    facts = _preflight(settings)

    image_id = _image_id(settings, remote=False)
    if not image_id:
        raise DualSparkError(
            f"required image {settings.image!r} is not present locally; "
            "build it from this example's Dockerfile first"
        )
    _log(f"using prebuilt image {settings.image} ({image_id})")
    _sync_image(settings, image_id)

    bundle = _prepare_bundle(settings, image_id)
    bundle_digest = _sync_bundle(settings, bundle)
    preparation = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "peer": settings.peer_target,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": PRECISION},
        "profile": _profile(),
        "image": {"name": settings.image, "id": image_id},
        "bundle": {
            "primary_path": str(bundle),
            "peer_path": str(layout["remote_bundle"]),
            "sha256": bundle_digest,
            "bytes": bundle.stat().st_size,
        },
        **facts,
    }
    _write_json(Path(layout["preparation"]), preparation)
    _log("preparation complete")
    return preparation


def _require_prepared(settings: Settings) -> dict[str, Any]:
    layout = _layout(settings)
    preparation_path = Path(layout["preparation"])
    if not preparation_path.is_file():
        raise DualSparkError("prepared assets are missing; run the prepare action first")
    try:
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DualSparkError("preparation.json is invalid; run prepare again") from exc
    if (
        preparation.get("schema_version") != 1
        or preparation.get("peer") != settings.peer_target
        or preparation.get("model")
        != {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": PRECISION}
        or preparation.get("profile") != _profile()
        or preparation.get("image", {}).get("name") != settings.image
    ):
        raise DualSparkError("prepared asset provenance does not match this invocation")

    expected_image_id = str(preparation.get("image", {}).get("id", ""))
    if not expected_image_id or _image_id(settings, remote=False) != expected_image_id:
        raise DualSparkError("the primary image changed; run prepare again")
    if _image_id(settings, remote=True) != expected_image_id:
        raise DualSparkError("the peer image changed; run prepare again")

    bundle = Path(layout["bundle"])
    if not bundle.is_file() or bundle.stat().st_size == 0:
        raise DualSparkError(f"required bundle is missing: {bundle}")
    expected_digest = str(preparation.get("bundle", {}).get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise DualSparkError("preparation.json contains an invalid bundle digest")
    if _sha256(bundle) != expected_digest:
        raise DualSparkError("the primary bundle changed; run prepare again")
    if _remote_sha256(settings, str(layout["remote_bundle"])) != expected_digest:
        raise DualSparkError("the peer bundle changed; run prepare again")
    return preparation


def _container_exists(settings: Settings, name: str, *, remote: bool) -> bool:
    argv = ["docker", "inspect", name]
    completed = (
        _remote_run(settings, argv, check=False)
        if remote
        else _run(argv, check=False)
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = (completed.stderr or completed.stdout or "").strip()
    suffix = f": {detail[-500:]}" if detail else ""
    raise DualSparkError(
        f"could not determine whether container {name!r} exists{suffix}"
    )


def _container_state(settings: Settings, name: str, *, remote: bool) -> dict[str, Any]:
    argv = ["docker", "inspect", "--format", "{{json .State}}", name]
    completed = _remote_run(settings, argv) if remote else _run(argv)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DualSparkError(f"cannot parse container state for {name}") from exc


def _sample_memory_tracker(settings: Settings, tracker: MemoryTracker) -> None:
    paths = [
        f"{tracker.cgroup_path}/memory.current",
        f"{tracker.cgroup_path}/memory.peak",
        f"{tracker.cgroup_path}/memory.swap.current",
        f"{tracker.cgroup_path}/memory.swap.peak",
        f"{tracker.cgroup_path}/memory.events",
    ]
    argv = ["cat", "--", *paths]
    completed = (
        _remote_run(settings, argv) if tracker.remote else _run(argv)
    )
    lines = [line.strip() for line in completed.stdout.splitlines()]
    values = lines[:4]
    event_lines = lines[4:]
    if len(values) != 4 or any(
        not re.fullmatch(r"[0-9]+", value) for value in values
    ):
        raise DualSparkError(
            f"invalid cgroup-v2 memory counters for {tracker.container}"
        )
    if not event_lines or any(
        not re.fullmatch(r"[a-z_]+ [0-9]+", line) for line in event_lines
    ):
        raise DualSparkError(
            f"invalid cgroup-v2 memory events for {tracker.container}"
        )
    current_bytes, peak_bytes, swap_current_bytes, swap_peak_bytes = map(int, values)
    if peak_bytes < current_bytes or swap_peak_bytes < swap_current_bytes:
        raise DualSparkError(
            f"inconsistent cgroup-v2 memory counters for {tracker.container}"
        )
    events = "\n".join(event_lines) + "\n"
    if tracker.samples == 0:
        tracker.initial_events = events
    tracker.latest_events = events
    tracker.peak_bytes = max(tracker.peak_bytes, peak_bytes)
    tracker.swap_peak_bytes = max(tracker.swap_peak_bytes, swap_peak_bytes)
    tracker.records.append(
        (
            time.time_ns() // 1_000_000,
            current_bytes,
            peak_bytes,
            swap_current_bytes,
            swap_peak_bytes,
        )
    )
    tracker.samples += 1


def _memory_tracker(
    settings: Settings,
    container: str,
    *,
    remote: bool,
) -> MemoryTracker:
    pid_argv = ["docker", "inspect", "--format", "{{.State.Pid}}", container]
    completed = (
        _remote_run(settings, pid_argv) if remote else _run(pid_argv)
    )
    pid_text = completed.stdout.strip()
    if not re.fullmatch(r"[1-9][0-9]*", pid_text):
        raise DualSparkError(f"container is not running for memory monitoring: {container}")

    cgroup_argv = ["cat", "--", f"/proc/{pid_text}/cgroup"]
    completed = (
        _remote_run(settings, cgroup_argv) if remote else _run(cgroup_argv)
    )
    relative_paths = [
        line[3:]
        for line in completed.stdout.splitlines()
        if line.startswith("0::")
    ]
    if len(relative_paths) != 1:
        raise DualSparkError(f"cannot resolve cgroup-v2 path for {container}")
    relative = relative_paths[0]
    if (
        not relative.startswith("/")
        or relative == "/"
        or ".." in Path(relative).parts
        or "\x00" in relative
    ):
        raise DualSparkError(f"unsafe cgroup-v2 path for {container}")
    cgroup_path = f"/sys/fs/cgroup{relative.rstrip('/')}"
    counter_paths = (
        f"{cgroup_path}/memory.current",
        f"{cgroup_path}/memory.peak",
        f"{cgroup_path}/memory.swap.current",
        f"{cgroup_path}/memory.swap.peak",
        f"{cgroup_path}/memory.events",
    )
    for path in counter_paths:
        check_argv = ["test", "-r", path]
        checked = (
            _remote_run(settings, check_argv, check=False)
            if remote
            else _run(check_argv, check=False)
        )
        if checked.returncode != 0:
            raise DualSparkError(
                f"required cgroup-v2 memory counter is unavailable for {container}: {path}"
            )
    tracker = MemoryTracker(container=container, remote=remote, cgroup_path=cgroup_path)
    _sample_memory_tracker(settings, tracker)
    return tracker


def _write_memory_evidence(tracker: MemoryTracker, destination: Path) -> None:
    if tracker.samples == 0 or not tracker.records:
        raise DualSparkError(f"no cgroup memory samples for {tracker.container}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        "epoch_ms,container_current_bytes,container_peak_bytes,"
        "container_swap_current_bytes,container_swap_peak_bytes"
    ]
    rows.extend(",".join(str(value) for value in record) for record in tracker.records)
    evidence = {
        destination / "memory.csv": "\n".join(rows) + "\n",
        destination / "memory-events.initial.txt": tracker.initial_events,
        destination / "memory-events.txt": tracker.latest_events,
    }
    for path, text in evidence.items():
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def _write_memory_evidence_noexcept(
    tracker: MemoryTracker,
    destination: Path,
) -> None:
    try:
        _write_memory_evidence(tracker, destination)
    except Exception as exc:
        _log(
            f"WARNING: could not preserve cgroup memory evidence "
            f"for {tracker.container}: {exc}"
        )


def _capture_container(
    settings: Settings,
    name: str,
    destination: Path,
    *,
    remote: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    logs_argv = ["docker", "logs", name]
    inspect_argv = ["docker", "inspect", name]
    logs = (
        _remote_run(settings, logs_argv, check=False)
        if remote
        else _run(logs_argv, check=False)
    )
    inspect = (
        _remote_run(settings, inspect_argv, check=False)
        if remote
        else _run(inspect_argv, check=False)
    )
    combined = (logs.stdout or "") + (logs.stderr or "")
    (destination / "container.log").write_text(combined, encoding="utf-8")
    (destination / "inspect.json").write_text(
        inspect.stdout or "[]\n",
        encoding="utf-8",
    )


def _capture_container_noexcept(
    settings: Settings,
    name: str,
    destination: Path,
    *,
    remote: bool,
) -> None:
    try:
        _capture_container(settings, name, destination, remote=remote)
    except Exception as exc:  # Preserve the original run failure during cleanup.
        _log(f"WARNING: could not capture container {name}: {exc}")


def _remove_container_noexcept(settings: Settings, name: str, *, remote: bool) -> bool:
    argv = ["docker", "rm", "-f", name]
    try:
        completed = (
            _remote_run(settings, argv, check=False, timeout=30)
            if remote
            else _run(argv, check=False, timeout=30)
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            suffix = f": {detail[-500:]}" if detail else ""
            _log(
                f"WARNING: docker rm -f returned {completed.returncode} "
                f"for exact container {name}{suffix}"
            )
    except Exception as exc:
        _log(f"WARNING: could not remove container {name}: {exc}")
    try:
        exists = _container_exists(settings, name, remote=remote)
    except Exception as exc:
        _log(f"WARNING: could not verify removal of exact container {name}: {exc}")
        return False
    if exists:
        try:
            state = _container_state(settings, name, remote=remote)
            running = state.get("Running")
        except Exception:
            running = "unknown"
        _log(
            f"WARNING: exact container {name} still exists after cleanup "
            f"(running={running})"
        )
        return False
    return True


def _wait_for_rendezvous(settings: Settings, path: Path, container: str) -> None:
    deadline = time.monotonic() + min(180, settings.runtime_timeout)
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size == NCCL_ID_BYTES:
            return
        state = _container_state(settings, container, remote=False)
        if not state.get("Running", False):
            raise DualSparkError(
                "rank 0 exited before publishing the NCCL rendezvous"
            )
        time.sleep(1)
    raise DualSparkError("timed out waiting for the NCCL rendezvous")


def _transfer_rendezvous(settings: Settings, local_path: Path, remote_path: str) -> None:
    incoming = f"{remote_path}.incoming"
    _remote_run(settings, ["rm", "-f", "--", incoming, remote_path], check=False)
    copied = False
    try:
        _run(
            [*_scp_argv(settings), str(local_path), f"{settings.peer_target}:{incoming}"],
            timeout=120,
            capture=False,
        )
        size = _remote_run(
            settings,
            ["stat", "-c", "%s", "--", incoming],
        ).stdout.strip()
        if size != str(NCCL_ID_BYTES):
            raise DualSparkError("peer received an invalid NCCL rendezvous file")
        if _remote_sha256(settings, incoming) != _sha256(local_path):
            raise DualSparkError("peer NCCL rendezvous checksum does not match")
        _remote_run(settings, ["mv", "-T", "--", incoming, remote_path])
        copied = True
    finally:
        if not copied:
            _remote_run(settings, ["rm", "-f", "--", incoming], check=False)


def _wait_for_containers(
    settings: Settings,
    containers: Sequence[tuple[str, bool]],
    trackers: Mapping[str, MemoryTracker],
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + settings.runtime_timeout
    states: dict[str, dict[str, Any]] = {}
    finished_trackers: set[str] = set()
    while time.monotonic() < deadline:
        states = {
            name: _container_state(settings, name, remote=remote)
            for name, remote in containers
        }
        for name, tracker in trackers.items():
            if name in finished_trackers:
                continue
            try:
                _sample_memory_tracker(settings, tracker)
            except DualSparkError as exc:
                if tracker.samples == 0:
                    raise
                if states[name].get("Running", False):
                    states[name] = _container_state(
                        settings,
                        name,
                        remote=tracker.remote,
                    )
                if states[name].get("Running", False):
                    raise
                _log(
                    f"WARNING: final cgroup memory sample was unavailable "
                    f"for {name}: {exc}"
                )
                finished_trackers.add(name)
        if all(not state.get("Running", False) for state in states.values()):
            return states
        if any(
            not state.get("Running", False) and int(state.get("ExitCode", 1)) != 0
            for state in states.values()
        ):
            return states
        time.sleep(1)
    raise DualSparkError("timed out waiting for Cosmos3 inference")


def _verify_frames(frames_dir: Path) -> None:
    expected = {f"frame_{index:04d}.png" for index in range(FRAME_COUNT)}
    frame_paths = tuple(frames_dir.glob("frame_*.png"))
    actual = {path.name for path in frame_paths if path.is_file()}
    if actual != expected or any(path.stat().st_size == 0 for path in frame_paths):
        raise DualSparkError(
            "rank 0 did not produce exactly 189 nonempty sequential frames"
        )


def _verify_rank1_has_no_frames(settings: Settings, remote_output: str) -> None:
    result = _remote_run(
        settings,
        [
            "find",
            remote_output,
            "-maxdepth",
            "2",
            "-type",
            "f",
            "-name",
            "frame_*.png",
            "-print",
            "-quit",
        ],
    )
    if result.stdout.strip():
        raise DualSparkError("rank 1 unexpectedly wrote video frames")


def _remove_rank1_empty_frames(settings: Settings, remote_output: str) -> None:
    frames = f"{remote_output}/frames"
    removed = _remote_run(settings, ["rmdir", "--", frames], check=False)
    if removed.returncode != 0:
        detail = (removed.stderr or removed.stdout or "").strip()
        suffix = f": {detail[-500:]}" if detail else ""
        raise DualSparkError(
            "could not remove rank 1's verified-empty frames directory"
            f"{suffix}"
        )


def _validate_rank_evidence(scene_paths: Mapping[str, Any]) -> None:
    for rank, directory_key in ((0, "rank0"), (1, "rank1_capture")):
        directory = Path(scene_paths[directory_key])
        log_path = directory / "container.log"
        inspect_path = directory / "inspect.json"
        if not log_path.is_file() or not inspect_path.is_file():
            raise DualSparkError(f"rank {rank} diagnostic evidence is incomplete")
        log_text = log_path.read_text(encoding="utf-8")
        has_ib_transport = any(
            re.search(pattern, log_text)
            for pattern in (
                r"Using network IB",
                r"via NET/IB",
                r"NET/IB[^\n]*Using",
            )
        )
        if not has_ib_transport:
            raise DualSparkError(f"rank {rank} did not report positive NCCL NET/IB evidence")
        if any(
            marker in log_text
            for marker in (
                "NET/Socket",
                "Using network Socket",
                "Failed to initialize NET plugin IB",
            )
        ):
            raise DualSparkError(f"rank {rank} fell back to NCCL socket transport")
        try:
            inspected = json.loads(inspect_path.read_text(encoding="utf-8"))
            state = inspected[0]["State"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DualSparkError(f"rank {rank} Docker state evidence is invalid") from exc
        if state.get("OOMKilled") is not False:
            raise DualSparkError(f"rank {rank} was OOM-killed or has unknown OOM state")
        if int(state.get("ExitCode", 1)) != 0:
            raise DualSparkError(f"rank {rank} did not exit successfully")


_COSMOS3_PERF_MS_FIELDS = (
    "prompt_conditioning_ms",
    "denoise_prep_ms",
    "denoiser_engine_load_ms",
    "denoiser_ms",
    "scheduler_cfg_ms",
    "vae_decoder_ms",
    "generation_excluding_denoiser_load_ms",
    "total_ms",
)


def _parse_cosmos3_performance(
    log_path: Path,
    scene: Scene,
    rank0_wall_seconds: float,
) -> dict[str, int | float]:
    marker = "[cosmos3.perf] "
    records = [
        line.split(marker, 1)[1].strip()
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if marker in line
    ]
    if len(records) != 1:
        raise DualSparkError(
            f"rank 0 must emit exactly one final [cosmos3.perf] record; found {len(records)}"
        )
    fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", records[0]))
    required = {*_COSMOS3_PERF_MS_FIELDS, "cp_size", "seed"}
    if not required.issubset(fields):
        missing = sorted(required - fields.keys())
        raise DualSparkError(f"rank-0 Cosmos3 performance record is missing {missing}")

    performance: dict[str, int | float] = {}
    for name in _COSMOS3_PERF_MS_FIELDS:
        try:
            value = float(fields[name])
        except ValueError as exc:
            raise DualSparkError(
                f"rank-0 Cosmos3 performance field {name} is not numeric"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise DualSparkError(
                f"rank-0 Cosmos3 performance field {name} is invalid"
            )
        performance[name] = value
    try:
        cp_size = int(fields["cp_size"])
        seed = int(fields["seed"])
    except ValueError as exc:
        raise DualSparkError("rank-0 Cosmos3 cp_size or seed is not an integer") from exc
    if cp_size != CP_SIZE or seed != scene.seed:
        raise DualSparkError(
            "rank-0 Cosmos3 performance record does not match the requested CP size and seed"
        )
    if not math.isfinite(rank0_wall_seconds) or rank0_wall_seconds < 0:
        raise DualSparkError("rank-0 wall-clock measurement is invalid")
    performance.update(
        {
            "cp_size": cp_size,
            "seed": seed,
            "inference_seconds": performance["total_ms"] / 1000.0,
            "total_seconds": performance["total_ms"] / 1000.0,
            "rank0_wall_seconds": rank0_wall_seconds,
        }
    )
    return performance


def _package_video(
    settings: Settings,
    frames_dir: Path,
    output: Path,
) -> dict[str, Any]:
    layout = _layout(settings)
    run_root = Path(layout["run_root"])
    _verify_frames(frames_dir)
    frames_in_container = "/run/" + str(frames_dir.relative_to(run_root))
    if output.exists():
        raise DualSparkError(f"refusing to overwrite an existing MP4: {output}")
    pending = output.with_name(f"{output.stem}.pending.mp4")
    if pending.exists():
        pending.unlink()
    pending_in_container = "/run/" + str(pending.relative_to(run_root))
    try:
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,src={run_root},dst=/run",
                "--entrypoint",
                "ffmpeg",
                settings.image,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(FRAME_RATE),
                "-start_number",
                "0",
                "-i",
                f"{frames_in_container}/frame_%04d.png",
                "-frames:v",
                str(FRAME_COUNT),
                "-r",
                str(FRAME_RATE),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                pending_in_container,
            ],
            timeout=20 * 60,
            capture=False,
        )
        probe = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,src={run_root},dst=/run,readonly",
                "--entrypoint",
                "ffprobe",
                settings.image,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                pending_in_container,
            ],
            timeout=120,
        )
        try:
            metadata = json.loads(probe.stdout)
            stream = metadata["streams"][0]
            duration = float(metadata["format"]["duration"])
            width = int(stream["width"])
            height = int(stream["height"])
            frames = int(stream["nb_read_frames"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DualSparkError("ffprobe returned an invalid video record") from exc
        if (
            stream.get("codec_name") != "h264"
            or width != WIDTH
            or height != HEIGHT
            or stream.get("r_frame_rate") != f"{FRAME_RATE}/1"
            or frames != FRAME_COUNT
            or not math.isclose(duration, FRAME_COUNT / FRAME_RATE, abs_tol=0.05)
        ):
            raise DualSparkError("packaged MP4 does not match the native 720p contract")
        os.replace(pending, output)
    except BaseException:
        if pending.exists():
            pending.unlink()
        raise
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "video": {
            "codec": "h264",
            "width": width,
            "height": height,
            "frames": frames,
            "fps": FRAME_RATE,
            "pixel_format": "yuv420p",
            "encoder": {"crf": 18, "preset": "slow", "faststart": True},
        },
    }


def _prepare_scene_directories(settings: Settings, scene: Scene) -> None:
    paths = _scene_layout(settings, scene)
    Path(paths["rank0"]).mkdir(parents=True, exist_ok=False)
    Path(paths["rank1_capture"]).mkdir(parents=True, exist_ok=False)
    Path(paths["rendezvous"]).parent.mkdir(parents=True, exist_ok=False)
    remote_scene_root = str(Path(paths["remote_rank1"]).parent)
    _remote_run(
        settings,
        [
            "install",
            "-d",
            "-m",
            "0700",
            remote_scene_root,
            str(paths["remote_rank1"]),
            str(Path(paths["remote_rendezvous"]).parent),
        ],
    )


def _run_scene(
    settings: Settings,
    scene: Scene,
    roce_hosts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    paths = _scene_layout(settings, scene)
    rank0_name = str(paths["rank0_container"])
    rank1_name = str(paths["rank1_container"])
    try:
        primary_uverbs = roce_hosts["primary"]["uverbs_device"]
        peer_uverbs = roce_hosts["peer"]["uverbs_device"]
    except KeyError as exc:
        raise DualSparkError("preflight did not resolve both uverbs devices") from exc
    for device in (primary_uverbs, peer_uverbs):
        if not re.fullmatch(r"/dev/infiniband/uverbs[0-9]+", device):
            raise DualSparkError(f"preflight returned an invalid uverbs device: {device}")

    if _container_exists(settings, rank0_name, remote=False):
        raise DualSparkError(f"primary container already exists: {rank0_name}")
    if _container_exists(settings, rank1_name, remote=True):
        raise DualSparkError(f"peer container already exists: {rank1_name}")
    _prepare_scene_directories(settings, scene)

    rank0_launch_attempted = False
    rank1_launch_attempted = False
    rank0_wall_started = 0.0
    rank0_wall_seconds = 0.0
    trackers: dict[str, MemoryTracker] = {}
    cleanup_failures: list[str] = []
    try:
        _log(f"starting {scene.slug}: primary rank 0")
        rank0_launch_attempted = True
        rank0_wall_started = time.monotonic()
        _run(
            _rank_docker_argv(
                settings, scene, 0, uverbs_device=primary_uverbs
            )
        )
        _wait_for_rendezvous(
            settings,
            Path(paths["rendezvous"]),
            rank0_name,
        )
        _transfer_rendezvous(
            settings,
            Path(paths["rendezvous"]),
            str(paths["remote_rendezvous"]),
        )
        _log(f"starting {scene.slug}: peer rank 1")
        rank1_launch_attempted = True
        _remote_run(
            settings,
            _rank_docker_argv(
                settings, scene, 1, uverbs_device=peer_uverbs
            ),
        )
        trackers = {
            rank0_name: _memory_tracker(settings, rank0_name, remote=False),
            rank1_name: _memory_tracker(settings, rank1_name, remote=True),
        }
        states = _wait_for_containers(
            settings,
            ((rank0_name, False), (rank1_name, True)),
            trackers,
        )
        rank0_wall_seconds = time.monotonic() - rank0_wall_started
        failures = {
            name: int(state.get("ExitCode", 1))
            for name, state in states.items()
            if int(state.get("ExitCode", 1)) != 0
        }
        if failures:
            raise DualSparkError(f"Cosmos3 ranks failed for {scene.slug}: {failures}")
    finally:
        launched = (
            (rank0_name, False, rank0_launch_attempted, Path(paths["rank0"])),
            (rank1_name, True, rank1_launch_attempted, Path(paths["rank1_capture"])),
        )
        for name, remote, attempted, destination in launched:
            if not attempted:
                continue
            try:
                exists = _container_exists(settings, name, remote=remote)
            except Exception as exc:
                exists = False
                _log(f"WARNING: could not probe exact container {name}: {exc}")
            if exists:
                _capture_container_noexcept(
                    settings, name, destination, remote=remote
                )
            tracker = trackers.get(name)
            if tracker is not None:
                _write_memory_evidence_noexcept(tracker, destination)
        for name, remote, attempted, _destination in launched:
            if attempted and not _remove_container_noexcept(
                settings, name, remote=remote
            ):
                cleanup_failures.append(name)

    if cleanup_failures:
        raise DualSparkError(
            f"could not verify cleanup of exact containers: {cleanup_failures}"
        )
    if set(trackers) != {rank0_name, rank1_name}:
        raise DualSparkError("memory telemetry is incomplete for the two ranks")
    for destination in (Path(paths["rank0"]), Path(paths["rank1_capture"])):
        for name in ("memory.csv", "memory-events.initial.txt", "memory-events.txt"):
            evidence_path = destination / name
            if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                raise DualSparkError(
                    f"cgroup memory evidence is incomplete: {evidence_path}"
                )

    _validate_rank_evidence(paths)
    performance = _parse_cosmos3_performance(
        Path(paths["rank0"]) / "container.log", scene, rank0_wall_seconds
    )
    memory = {
        "primary": {
            "rank": 0,
            "peak_bytes": trackers[rank0_name].peak_bytes,
            "swap_peak_bytes": trackers[rank0_name].swap_peak_bytes,
            "samples": trackers[rank0_name].samples,
            "evidence": {
                "samples_csv": str(Path(paths["rank0"]) / "memory.csv"),
                "initial_events": str(Path(paths["rank0"]) / "memory-events.initial.txt"),
                "final_events": str(Path(paths["rank0"]) / "memory-events.txt"),
                "container_state": str(Path(paths["rank0"]) / "inspect.json"),
            },
            "source": "cgroup-v2 memory.peak",
        },
        "peer": {
            "rank": 1,
            "peak_bytes": trackers[rank1_name].peak_bytes,
            "swap_peak_bytes": trackers[rank1_name].swap_peak_bytes,
            "samples": trackers[rank1_name].samples,
            "evidence": {
                "samples_csv": str(Path(paths["rank1_capture"]) / "memory.csv"),
                "initial_events": str(Path(paths["rank1_capture"]) / "memory-events.initial.txt"),
                "final_events": str(Path(paths["rank1_capture"]) / "memory-events.txt"),
                "container_state": str(Path(paths["rank1_capture"]) / "inspect.json"),
            },
            "source": "cgroup-v2 memory.peak",
        },
    }
    _verify_rank1_has_no_frames(settings, str(paths["remote_rank1"]))
    _remove_rank1_empty_frames(settings, str(paths["remote_rank1"]))
    artifact = _package_video(
        settings,
        Path(paths["rank0"]) / "frames",
        Path(paths["mp4"]),
    )
    _log(f"completed {scene.slug}: {paths['mp4']}")
    return {
        "slug": scene.slug,
        "seed": scene.seed,
        "prompt": _prompt_record(scene),
        "rank_hosts": {"0": socket.gethostname(), "1": settings.peer_host},
        "rendezvous_name": Path(paths["rendezvous"]).name,
        "performance": performance,
        "memory": memory,
        "artifact": artifact,
    }


def run_scenes(settings: Settings) -> dict[str, Any]:
    layout = _layout(settings)
    facts = _preflight(settings)
    _ensure_private_remote_root(settings)
    preparation = _require_prepared(settings)
    if preparation.get("host_identity") != facts["host_identity"]:
        raise DualSparkError(
            "prepared assets belong to a different primary/peer host pair; run prepare again"
        )

    run_root = Path(layout["run_root"])
    remote_run_root = str(layout["remote_run_root"])
    if run_root.exists():
        raise DualSparkError(f"run directory already exists: {run_root}")
    if _remote_run(settings, ["test", "-e", remote_run_root], check=False).returncode == 0:
        raise DualSparkError(f"peer run directory already exists: {remote_run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    _remote_run(settings, ["install", "-d", "-m", "0700", remote_run_root])

    artifacts = []
    for scene in _selected_scenes(settings):
        artifacts.append(_run_scene(settings, scene, facts["roce_hosts"]))

    run_record = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": settings.run_id,
        "scene_selection": settings.scene,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": PRECISION},
        "profile": _profile(),
        "hosts": {
            "primary": socket.gethostname(),
            "peer": settings.peer_host,
            "primary_gpu": preparation["primary_gpu"],
            "peer_gpu": preparation["peer_gpu"],
            "identity": facts["host_identity"],
        },
        "artifacts": {
            "image": preparation["image"],
            "bundle": preparation["bundle"],
        },
        "roce": {
            "hca": settings.ib_hca,
            "gid_index": settings.gid_index,
            "network_interface": settings.net_iface,
            "address_family": "AF_INET",
            "hosts": facts["roce_hosts"],
        },
        "scenes": artifacts,
    }
    _write_json(Path(layout["run_json"]), run_record)
    _log(f"all scenes complete; provenance: {layout['run_json']}")
    return run_record


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = _settings(_parser().parse_args(argv))
        if settings.dry_run:
            json.dump(execution_plan(settings), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0

        if settings.action in {"prepare", "all"}:
            prepare(settings)
        if settings.action in {"run", "all"}:
            run_scenes(settings)
        return 0
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except DualSparkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
