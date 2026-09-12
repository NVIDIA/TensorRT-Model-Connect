/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const fs = require('fs');
const path = require('path');

const TASKS = {
  audio_generation: ['Text to speech', 'Audio', 'text-to-speech', 'generate-audio'],
  classification: ['Image classification', 'Computer Vision', 'image-classification', 'classify'],
  embedding: ['Embedding', 'Natural Language Processing', 'feature-extraction', 'embed'],
  encoding: ['Encoding', 'Natural Language Processing', 'feature-extraction', 'encode'],
  image_edit: ['Image editing', 'Computer Vision', 'image-to-image', 'generate-image'],
  image_features: ['Image feature extraction', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  image_generation: ['Image generation', 'Computer Vision', 'text-to-image', 'generate-image'],
  image_generation_batch: ['Batch image generation', 'Computer Vision', 'text-to-image', 'generate-image-batch'],
  monocular_geometry: ['Monocular geometry', 'Computer Vision', 'depth-estimation', 'geometry'],
  object_detection: ['Object detection', 'Computer Vision', 'object-detection', 'detect'],
  pose_hypothesis_refinement: ['Pose hypothesis refinement', 'Robotics', 'robotics', null],
  prompted_segmentation: ['Prompted segmentation', 'Computer Vision', 'mask-generation', 'segment'],
  reranking: ['Text ranking', 'Natural Language Processing', 'text-ranking', 'rerank'],
  robot_control: ['Robot control', 'Robotics', 'robotics', 'control'],
  segmentation: ['Image segmentation', 'Computer Vision', 'image-segmentation', 'segment'],
  speech_session: ['Speech session', 'Audio', 'audio-to-audio', 'speak'],
  speech_to_speech: ['Speech to speech', 'Audio', 'audio-to-audio', 'speak'],
  stereo_disparity: ['Depth estimation', 'Computer Vision', 'depth-estimation', 'disparity'],
  structure_prediction: ['Structure prediction', 'Biology', 'protein-folding', null],
  text_generation: ['Text generation', 'Natural Language Processing', 'text-generation', 'run'],
  text_prompted_segmentation: ['Text-prompted segmentation', 'Computer Vision', 'mask-generation', 'segment'],
  time_series_forecast: ['Time-series forecasting', 'Time Series', 'time-series-forecasting', 'forecast'],
  transcription: ['Speech recognition', 'Audio', 'automatic-speech-recognition', 'transcribe'],
  transcription_streaming: ['Streaming speech recognition', 'Audio', 'automatic-speech-recognition', 'transcribe'],
  video_segmentation: ['Video segmentation', 'Computer Vision', 'image-segmentation', 'video-segment'],
  vision_language_generation: ['Vision-language generation', 'Multimodal', 'image-text-to-text', 'run'],
  world_model_generation: ['World-model generation', 'Computer Vision', 'image-to-video', 'generate-world'],

  // Semantic SDK contracts. A null taxonomy or command means no matching
  // Hugging Face task page or implemented CLI workload is advertised.
  text_continuation: ['Text continuation', 'Natural Language Processing', 'text-generation', 'run'],
  conditional_text_generation: ['Conditional text generation', 'Natural Language Processing', 'text-generation', 'run'],
  corrupted_text_reconstruction: ['Corrupted text reconstruction', 'Natural Language Processing', null, 'run'],
  unconditional_text_generation: ['Unconditional text generation', 'Natural Language Processing', 'text-generation', 'run'],
  text_translation: ['Text translation', 'Natural Language Processing', 'translation', 'run'],
  text_summarization: ['Text summarization', 'Natural Language Processing', 'summarization', 'run'],
  text_prefix_suffix_infilling: ['Text prefix/suffix infilling', 'Natural Language Processing', null, 'run'],
  context_question_answering: ['Context question answering', 'Natural Language Processing', 'question-answering', 'run'],
  batch_text_continuation: ['Batch text continuation', 'Natural Language Processing', 'text-generation', 'run'],
  streaming_text_continuation: ['Streaming text continuation', 'Natural Language Processing', 'text-generation', 'run'],
  streaming_images_text_to_text: ['Streaming images and text to text', 'Multimodal', 'image-text-to-text', null],
  streaming_text_conversation: ['Streaming text conversation', 'Natural Language Processing', 'text-generation', null],
  streaming_video_text_to_text: ['Streaming video and text to text', 'Multimodal', 'video-text-to-text', null],
  streaming_image_video_text_to_text: ['Streaming image, video and text to text', 'Multimodal', null, null],
  streaming_images_text_conversation: ['Streaming images and text conversation', 'Multimodal', 'image-text-to-text', null],
  streaming_video_text_conversation: ['Streaming video and text conversation', 'Multimodal', 'video-text-to-text', null],

  text_query_documents_to_relevance: ['Query and documents to relevance', 'Natural Language Processing', 'text-ranking', 'rerank'],
  text_to_token_features: ['Text token features', 'Natural Language Processing', 'feature-extraction', 'encode'],
  text_pair_to_token_features: ['Text-pair token features', 'Natural Language Processing', 'feature-extraction', 'encode'],
  text_to_pooled_features: ['Pooled text features', 'Natural Language Processing', 'feature-extraction', 'encode'],
  text_to_embedding: ['Text embedding', 'Natural Language Processing', 'feature-extraction', 'embed'],
  title_body_to_embedding: ['Title and body embedding', 'Natural Language Processing', 'feature-extraction', 'embed'],
  masked_text_to_token_scores: ['Masked text token scores', 'Natural Language Processing', 'fill-mask', 'encode'],
  text_pair_to_pretraining_relation_scores: ['Text-pair pretraining relation scores', 'Natural Language Processing', null, 'encode'],
  text_to_replaced_token_scores: ['Replaced-token scores', 'Natural Language Processing', null, 'encode'],
  text_prediction_positions_to_token_scores: ['Selected-position token scores', 'Natural Language Processing', null, 'encode'],
  image_to_token_features: ['Image token features', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  image_to_spatial_features: ['Spatial image features', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  image_to_pooled_features: ['Pooled image features', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  image_to_embedding: ['Image embedding', 'Computer Vision', 'image-feature-extraction', 'embed'],
  image_text_to_embedding: ['Image and text embedding', 'Multimodal', null, 'embed'],
  text_pair_to_relevance: ['Text-pair relevance', 'Natural Language Processing', 'text-ranking', 'rerank'],
  text_image_to_relevance: ['Text and image relevance', 'Multimodal', null, 'rerank'],
  text_image_text_to_relevance: ['Text, image and text relevance', 'Multimodal', null, 'rerank'],
  image_to_class_scores: ['Image class scores', 'Computer Vision', 'image-classification', 'classify'],
  image_to_token_and_pooled_features: ['Image token and pooled features', 'Computer Vision', 'image-feature-extraction', 'extract-features'],
  batch_image_to_class_scores: ['Batch image class scores', 'Computer Vision', 'image-classification', null],
  batch_image_to_token_features: ['Batch image token features', 'Computer Vision', 'image-feature-extraction', null],
  batch_image_to_spatial_features: ['Batch spatial image features', 'Computer Vision', 'image-feature-extraction', null],
  batch_image_to_pooled_features: ['Batch pooled image features', 'Computer Vision', 'image-feature-extraction', null],
  batch_text_to_embedding: ['Batch text embedding', 'Natural Language Processing', 'feature-extraction', null],
  batch_text_to_token_features: ['Batch text token features', 'Natural Language Processing', 'feature-extraction', null],
  batch_image_to_token_and_pooled_features: ['Batch image token and pooled features', 'Computer Vision', 'image-feature-extraction', null],

  text_to_image: ['Text to image', 'Computer Vision', 'text-to-image', 'generate-image'],
  images_text_to_image_edit: ['Images and text to edited image', 'Multimodal', 'image-text-to-image', 'generate-image'],
  masked_image_text_to_image: ['Masked image and text to image', 'Computer Vision', 'image-to-image', 'generate-image'],
  batch_text_to_image: ['Batch text to image', 'Computer Vision', 'text-to-image', 'generate-image-batch'],

  text_to_audio: ['Text to audio', 'Audio', null, 'generate-audio'],
  text_audio_token_history_to_audio: ['Text and audio-token history to audio', 'Audio', null, null],
  batch_text_audio_token_history_to_audio: ['Batch text and audio-token history to audio', 'Audio', null, null],
  text_to_speech: ['Text to speech', 'Audio', 'text-to-speech', 'generate-audio'],
  speech_transcription: ['Speech transcription', 'Audio', 'automatic-speech-recognition', 'transcribe'],
  speech_translation: ['Speech translation', 'Audio', null, 'transcribe'],
  audio_language_identification: ['Audio language identification', 'Audio', 'audio-classification', null],
  speech_to_speech_response: ['Speech-to-speech response', 'Audio', 'audio-to-audio', 'speak'],
  batch_speech_transcription: ['Batch speech transcription', 'Audio', 'automatic-speech-recognition', 'transcribe-batch'],
  batch_speech_translation: ['Batch speech translation', 'Audio', null, 'transcribe-batch'],
  mixed_batch_speech_to_text: ['Mixed batch speech to text', 'Audio', null, 'transcribe-batch'],
  batch_text_to_audio: ['Batch text to audio', 'Audio', null, null],
  batch_text_to_speech: ['Batch text to speech', 'Audio', 'text-to-speech', null],
  streaming_speech_transcription: ['Streaming speech transcription', 'Audio', 'automatic-speech-recognition', 'transcribe-streaming'],
  streaming_text_to_speech: ['Streaming text to speech', 'Audio', 'text-to-speech', 'generate-audio'],
  duplex_speech_dialogue: ['Duplex speech dialogue', 'Audio', null, 'speech-session'],
  offline_speech_dialogue: ['Offline speech dialogue', 'Audio', null, 'speech-session'],
  tool_speech_dialogue: ['Speech dialogue with tools', 'Audio', null, null],

  images_text_to_text: ['Images and text to text', 'Multimodal', 'image-text-to-text', 'run'],
  video_text_to_text: ['Video and text to text', 'Multimodal', 'video-text-to-text', null],
  image_video_text_to_text: ['Image, video and text to text', 'Multimodal', null, null],
  audio_text_to_text: ['Audio and text to text', 'Multimodal', 'audio-text-to-text', null],
  image_audio_to_text: ['Image and audio to text', 'Multimodal', null, null],
  audio_video_text_to_text: ['Audio, video and text to text', 'Multimodal', null, null],
  image_audio_text_to_text: ['Image, audio and text to text', 'Multimodal', null, null],
  image_audio_text_to_text_speech_response: ['Image, audio and text to text and speech', 'Multimodal', null, null],
  text_conversation: ['Text conversation', 'Natural Language Processing', 'text-generation', null],
  images_text_conversation: ['Images and text conversation', 'Multimodal', 'image-text-to-text', null],
  video_text_conversation: ['Video and text conversation', 'Multimodal', 'video-text-to-text', null],
  batch_text_conversation: ['Batch text conversation', 'Natural Language Processing', 'text-generation', null],
  batch_video_text_conversation: ['Batch video and text conversation', 'Multimodal', 'video-text-to-text', null],
  batch_audio_text_conversation: ['Batch audio and text conversation', 'Multimodal', 'audio-text-to-text', null],
  batch_image_audio_text_conversation: ['Batch image, audio and text conversation', 'Multimodal', null, null],
  batch_text_images_video_conversations: ['Batch text, image or video conversations', 'Multimodal', null, null],
  batch_text_images_audio_conversations: ['Batch text, image or audio conversations', 'Multimodal', null, null],
  batch_images_text_conversation: ['Batch images and text conversation', 'Multimodal', 'image-text-to-text', null],
  text_label_classification: ['Text label classification', 'Natural Language Processing', 'text-classification', null],
  text_pair_label_classification: ['Text-pair label classification', 'Natural Language Processing', 'text-classification', null],
  text_encoder_decoder_hidden_states: ['Text encoder-decoder hidden states', 'Natural Language Processing', 'feature-extraction', null],

  series_to_point_forecast: ['Point forecasting', 'Time Series', null, 'forecast'],
  series_to_quantile_forecast: ['Quantile forecasting', 'Time Series', null, 'forecast'],
  series_to_point_and_quantile_forecast: ['Point and quantile forecasting', 'Time Series', null, 'forecast'],
  series_to_regression_distribution: ['Series regression distribution', 'Time Series', null, 'forecast'],
  latent_conditioned_text_generation: ['Latent-conditioned text generation', 'Natural Language Processing', null, 'run'],
  latent_replay_to_text: ['Latent replay to text', 'Natural Language Processing', null, 'run'],
  latent_denoising_step: ['Latent denoising step', 'Natural Language Processing', null, 'solve'],
  latent_to_token_logits: ['Latents to token logits', 'Natural Language Processing', null, 'solve'],
  batch_series_to_point_forecast: ['Batch point forecasting', 'Time Series', null, null],
  batch_series_to_quantile_forecast: ['Batch quantile forecasting', 'Time Series', null, null],
  batch_series_to_point_and_quantile_forecast: ['Batch point and quantile forecasting', 'Time Series', null, null],

  image_to_semantic_segmentation: ['Semantic image segmentation', 'Computer Vision', 'image-segmentation', 'segment'],
  image_points_to_masks: ['Image and points to masks', 'Computer Vision', 'mask-generation', 'segment'],
  image_box_to_masks: ['Image and box to masks', 'Computer Vision', 'mask-generation', null],
  image_mask_to_masks: ['Image and prior mask to masks', 'Computer Vision', 'mask-generation', null],
  image_to_mask_proposals: ['Image mask proposals', 'Computer Vision', 'mask-generation', 'segment'],
  image_text_to_instance_masks: ['Image and text to instance masks', 'Computer Vision', 'mask-generation', 'segment-prompted'],
  image_box_exemplars_to_instance_masks: ['Image and box exemplars to instance masks', 'Computer Vision', 'mask-generation', null],
  stereo_images_to_disparity: ['Stereo images to disparity', 'Computer Vision', 'depth-estimation', 'disparity'],
  image_to_metric_geometry: ['Metric image geometry', 'Computer Vision', 'depth-estimation', 'geometry'],
  image_to_boxes: ['Image object detection', 'Computer Vision', 'object-detection', 'detect'],
  molecular_document_to_structure: ['Molecular document to structure', 'Molecular Modeling', null, 'predict-structure'],
  image_text_to_boxes: ['Image and text to boxes', 'Multimodal', 'zero-shot-object-detection', 'detect'],
  image_text_to_points: ['Image and text to points', 'Multimodal', null, 'detect'],
  pose_hypotheses_crops_to_refined_poses: ['Pose hypotheses and crops to refined poses', 'Robotics', null, null],
  rgbd_mesh_mask_to_object_pose: ['RGB-D, mesh and mask to object pose', 'Robotics', null, null],
  batch_image_text_to_boxes: ['Batch image and text to boxes', 'Multimodal', 'zero-shot-object-detection', null],

  frames_to_detected_mask_tracks: ['Frames to detected mask tracks', 'Computer Vision', null, 'video-segment'],
  frames_text_to_mask_tracks: ['Frames and text to mask tracks', 'Multimodal', null, 'video-segment'],
  prompt_frame_text_to_mask_tracks: ['Prompt frame and text to mask tracks', 'Multimodal', null, 'video-segment'],
  interactive_image_masks: ['Interactive image masks', 'Computer Vision', 'mask-generation', null],
  frames_points_to_mask_tracks: ['Frames and points to mask tracks', 'Computer Vision', null, null],
  frames_box_to_mask_tracks: ['Frames and box to mask tracks', 'Computer Vision', null, null],
  frames_mask_to_mask_tracks: ['Frames and mask to mask tracks', 'Computer Vision', null, null],
  interactive_frames_text_to_mask_tracks: ['Interactive frames and text to mask tracks', 'Multimodal', null, null],
  frames_box_exemplar_to_mask_tracks: ['Frames and box exemplar to mask tracks', 'Computer Vision', null, null],
  crop_pose_tracking: ['Crop-based pose tracking', 'Robotics', null, null],
  rgbd_initialized_pose_to_tracked_pose: ['RGB-D and initialized pose to tracked pose', 'Robotics', null, null],
  image_state_to_action_chunk: ['Image and state to action chunk', 'Robotics', null, 'control'],
  image_state_action_queue: ['Image and state action queue', 'Robotics', null, null],

  recurrent_tokens_to_logits: ['Recurrent tokens to logits', 'Natural Language Processing', null, null],
  recurrent_embeddings_to_logits: ['Recurrent embeddings to logits', 'Natural Language Processing', null, null],
  recurrent_tokens_to_hidden_states: ['Recurrent tokens to hidden states', 'Natural Language Processing', null, null],
  recurrent_embeddings_to_hidden_states: ['Recurrent embeddings to hidden states', 'Natural Language Processing', null, null],
  batch_recurrent_tokens_to_logits: ['Batch recurrent tokens to logits', 'Natural Language Processing', null, null],
  batch_recurrent_embeddings_to_logits: ['Batch recurrent embeddings to logits', 'Natural Language Processing', null, null],
  batch_recurrent_tokens_to_hidden_states: ['Batch recurrent tokens to hidden states', 'Natural Language Processing', null, null],
  batch_recurrent_embeddings_to_hidden_states: ['Batch recurrent embeddings to hidden states', 'Natural Language Processing', null, null],

  text_to_video: ['Text to video', 'Computer Vision', 'text-to-video', 'generate-video'],
  initial_image_text_to_video: ['Initial image and text to video', 'Multimodal', 'image-text-to-video', 'generate-video'],
  boundary_frames_text_to_video: ['Boundary frames and text to video', 'Multimodal', 'image-text-to-video', null],
  timed_frames_text_to_video: ['Timed frames and text to video', 'Multimodal', 'image-text-to-video', null],
  video_text_to_video_edit: ['Video and text to edited video', 'Computer Vision', 'video-to-video', null],
  masked_video_text_to_video: ['Masked video and text to video', 'Computer Vision', 'video-to-video', null],
  masked_video_reference_images_text_to_video: ['Masked video, reference images and text to video', 'Multimodal', null, null],
  image_text_action_to_video: ['Image, text and action to video', 'Multimodal', null, 'generate-world'],
  image_text_camera_trajectory_to_video: ['Image, text and camera trajectory to video', 'Multimodal', null, 'generate-world'],
  video_text_to_future_video: ['Video and text to future video', 'Multimodal', null, null],
  image_action_to_future_video: ['Image and action to future video', 'Multimodal', null, null],
  video_action_to_future_video: ['Video and action to future video', 'Multimodal', null, null],
  video_to_action_sequence: ['Video to action sequence', 'Robotics', null, null],
  image_to_action_and_video: ['Image to action and video', 'Robotics', null, null],
  video_to_action_and_video: ['Video to action and video', 'Robotics', null, null],
  text_to_audio_video: ['Text to audio and video', 'Multimodal', null, null],
  initial_image_text_to_audio_video: ['Initial image and text to audio and video', 'Multimodal', null, null],
  last_image_text_to_audio_video: ['Last image and text to audio and video', 'Multimodal', null, null],
  boundary_frames_text_to_audio_video: ['Boundary frames and text to audio and video', 'Multimodal', null, null],
  references_text_to_audio_video: ['References and text to audio and video', 'Multimodal', null, null],
  batch_text_to_video: ['Batch text to video', 'Computer Vision', 'text-to-video', null],
  batch_initial_image_text_to_video: ['Batch initial image and text to video', 'Multimodal', 'image-text-to-video', null],
  batch_video_text_to_future_video: ['Batch video and text to future video', 'Multimodal', null, null],
  batch_image_action_to_future_video: ['Batch image and action to future video', 'Multimodal', null, null],
  batch_video_to_action_sequence: ['Batch video to action sequence', 'Robotics', null, null],
  batch_image_to_action_and_video: ['Batch image to action and video', 'Robotics', null, null],
  batch_text_to_audio_video: ['Batch text to audio and video', 'Multimodal', null, null],
  batch_initial_image_text_to_audio_video: ['Batch initial image and text to audio and video', 'Multimodal', null, null],
};

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to parse ${filePath}: ${error.message}`);
  }
}

function familyDirectories(repoRoot) {
  const root = path.join(repoRoot, 'families');
  return fs.readdirSync(root, {withFileTypes: true})
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('_'))
    .map((entry) => ({name: entry.name, root: path.join(root, entry.name)}))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function requireFamilyShape(family) {
  for (const relative of ['support.py', 'model.py', 'runtime/CMakeLists.txt', 'tests']) {
    if (!fs.existsSync(path.join(family.root, relative))) {
      throw new Error(`${family.name} is missing families/${family.name}/${relative}`);
    }
  }
  if (fs.existsSync(path.join(family.root, 'MODEL.toml'))) {
    throw new Error(`${family.name} contains forbidden MODEL.toml metadata`);
  }
}

function manifestFiles(family) {
  const root = path.join(family.root, 'tests', 'manifests');
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root, {withFileTypes: true})
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function profileFromManifest(repoRoot, family, filePath) {
  const manifest = readJson(filePath);
  if (manifest.runtime_strategy !== undefined || manifest.task_strategy !== undefined) {
    throw new Error(`${filePath} uses a removed strategy field`);
  }
  if (!manifest.name || !manifest.bundle
      || manifest.family !== family.name || !manifest.task) {
    throw new Error(`${filePath} must declare name, bundle, exact family, and task`);
  }
  const task = Object.hasOwn(TASKS, manifest.task) ? TASKS[manifest.task] : undefined;
  if (!task) throw new Error(`${filePath} declares unknown task ${manifest.task}`);
  if (typeof manifest.precision !== 'string' || !Number.isInteger(manifest.tensor_parallel_size)) {
    throw new Error(`${filePath} must declare precision and tensor_parallel_size`);
  }
  const tensorParallelSize = manifest.tensor_parallel_size;
  const contextParallelSize = manifest.context_parallel_size ?? 1;
  if (tensorParallelSize < 1 || !Number.isInteger(contextParallelSize)
      || contextParallelSize < 1 || (tensorParallelSize > 1 && contextParallelSize > 1)) {
    throw new Error(`${filePath} has invalid or overlapping parallel sizes`);
  }
  const parallelSize = Math.max(tensorParallelSize, contextParallelSize);
  const parallelMode = contextParallelSize > 1
    ? 'context_parallel'
    : tensorParallelSize > 1 ? 'tensor_parallel' : 'single_device';
  return {
    profile: manifest.name,
    hfId: manifest.hf_id || 'prepared local checkpoint',
    revision: manifest.hf_id ? manifest.hf_revision || 'not pinned' : 'not applicable',
    bundle: manifest.bundle,
    family: family.name,
    task: manifest.task,
    taskSlugs: [manifest.task.replaceAll('_', '-')],
    cliCommands: task[3] ? [task[3]] : [],
    precision: manifest.precision,
    quantization: manifest.quantization || 'none',
    parallelMode,
    parallelSize,
    testcases: Array.isArray(manifest.testcases)
      ? manifest.testcases.map((testcase) => testcase.name).filter(Boolean)
      : [],
    fp32Layers: Array.isArray(manifest.fp32_layers) ? manifest.fp32_layers : [],
    sourcePath: path.relative(repoRoot, filePath).replaceAll('\\', '/'),
  };
}

function taskRecipe(taskName, profiles) {
  const [label, category, hfSlug] = TASKS[taskName];
  const slug = taskName.replaceAll('_', '-');
  const byFamily = new Map();
  for (const profile of profiles) {
    if (!byFamily.has(profile.family)) byFamily.set(profile.family, []);
    byFamily.get(profile.family).push(profile);
  }
  return {
    task: taskName,
    slug,
    label,
    category,
    description: `Family-owned implementations of the ${label.toLowerCase()} task interface.`,
    hfUrl: hfSlug ? `https://huggingface.co/tasks/${hfSlug}` : null,
    recipeCount: profiles.length,
    families: [...byFamily.entries()].map(([family, familyProfiles]) => ({
      family,
      slug: family.replaceAll('_', '-'),
      recipeCount: familyProfiles.length,
      hfIds: [...new Set(familyProfiles.map((profile) => profile.hfId))].sort(),
      cliCommands: [...new Set(familyProfiles.flatMap((profile) => profile.cliCommands))].sort(),
    })).sort((left, right) => left.family.localeCompare(right.family)),
  };
}

function collectModelSupportInventory(repoRoot) {
  const families = familyDirectories(repoRoot);
  const profiles = [];
  for (const family of families) {
    requireFamilyShape(family);
    for (const filePath of manifestFiles(family)) {
      profiles.push(profileFromManifest(repoRoot, family, filePath));
    }
  }
  profiles.sort((left, right) =>
    left.task.localeCompare(right.task) ||
    left.family.localeCompare(right.family) ||
    left.profile.localeCompare(right.profile));

  const familyRecipes = families.map((family) => {
    const owned = profiles.filter((profile) => profile.family === family.name);
    return {
      family: family.name,
      slug: family.name.replaceAll('_', '-'),
      profiles: owned,
      taskSlugs: [...new Set(owned.flatMap((profile) => profile.taskSlugs))].sort(),
      cliCommands: [...new Set(owned.flatMap((profile) => profile.cliCommands))].sort(),
    };
  });
  const profilesByTask = new Map();
  for (const profile of profiles) {
    if (!profilesByTask.has(profile.task)) profilesByTask.set(profile.task, []);
    profilesByTask.get(profile.task).push(profile);
  }
  const taskRecipes = [...profilesByTask.entries()]
    .map(([task, taskProfiles]) => taskRecipe(task, taskProfiles))
    .sort((left, right) => left.label.localeCompare(right.label));

  return {
    familyCount: families.length,
    familyNames: families.map((family) => family.name),
    manifestCount: profiles.length,
    modelProfiles: profiles,
    familyRecipes,
    taskRecipes,
  };
}

function modelSupportInventoryPlugin(context) {
  return {
    name: 'model-support-inventory',
    loadContent() {
      return collectModelSupportInventory(path.resolve(context.siteDir, '..'));
    },
    contentLoaded({content, actions}) {
      actions.setGlobalData(content);
      const routeBase = context.baseUrl.replace(/\/$/, '');
      const taskPage = path.join(context.siteDir, 'src/components/ModelRecipes/TaskPage.js');
      const familyPage = path.join(context.siteDir, 'src/components/ModelRecipes/FamilyPage.js');
      for (const task of content.taskRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/tasks/${task.slug}`,
          component: taskPage,
          exact: true,
          props: {taskSlug: task.slug},
        });
      }
      for (const family of content.familyRecipes) {
        actions.addRoute({
          path: `${routeBase}/models-recipes/model-recipes/families/${family.slug}`,
          component: familyPage,
          exact: true,
          props: {familySlug: family.slug},
        });
      }
    },
  };
}

module.exports = modelSupportInventoryPlugin;
module.exports.collectModelSupportInventory = collectModelSupportInventory;
