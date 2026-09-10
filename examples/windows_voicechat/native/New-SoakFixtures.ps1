# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [string]$Voice = 'Microsoft Zira Desktop'
)

$ErrorActionPreference = 'Stop'
if (-not $OutputDirectory) {
    $sourceDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
    $OutputDirectory = Join-Path (Split-Path $sourceDirectory -Parent) 'logs\voice-soak-fixtures'
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
Add-Type -AssemblyName System.Speech

# Different requests and explicit topic changes expose semantic loops that
# repeating the same successful question cannot detect. These are synthesized
# test inputs only; the application and the model receive ordinary microphone
# PCM through their production input path.
$turns = @(
    @{ id = '01-bedtime-story'; text = 'Please tell me a long bedtime story about a rabbit who explores a forest.'; expectedAny = @('rabbit', 'bunny'); interruptDuringReply = $true },
    @{ id = '02-stop-story-arithmetic'; text = 'Stop that story and tell me what seven plus five is.'; expectedAny = @('twelve', '12') },
    @{ id = '03-red-planet'; text = 'Which planet is known as the red planet?'; expectedAny = @('Mars') },
    @{ id = '04-japan-capital'; text = 'What is the capital city of Japan?'; expectedAny = @('Tokyo') },
    @{ id = '05-freezing-water'; text = 'What do we call water when it freezes solid?'; expectedAny = @('ice') },
    @{ id = '06-triangle'; text = 'How many sides does a triangle have?'; expectedAny = @('three', '3') },
    @{ id = '07-hot-opposite'; text = 'What is the opposite of hot?'; expectedAny = @('cold') },
    @{ id = '08-largest-ocean'; text = 'Which ocean is the largest ocean on Earth?'; expectedAny = @('Pacific') },
    @{ id = '09-keys-instrument'; text = 'Name the musical instrument with black and white keys that you play while sitting on a bench.'; expectedAny = @('piano') },
    @{ id = '10-egypt-continent'; text = 'On which continent is Egypt located?'; expectedAny = @('Africa', 'African') },
    @{ id = '11-week-days'; text = 'How many days are there in one week?'; expectedAny = @('seven', '7') },
    @{ id = '12-final-topic-change'; text = 'What color is a clear daytime sky?'; expectedAny = @('blue') }
)
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
try {
    $synthesizer.SelectVoice($Voice)
    $synthesizer.Rate = 0
    $synthesizer.Volume = 100
    foreach ($turn in $turns) {
        $audioPath = Join-Path $OutputDirectory ($turn.id + '.wav')
        $synthesizer.SetOutputToWaveFile($audioPath, $format)
        $synthesizer.Speak($turn.text)
        $synthesizer.SetOutputToNull()
        $turn.audio = $audioPath
        $turn.sha256 = (Get-FileHash -LiteralPath $audioPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($turn.id -ne '01-bedtime-story') {
            $turn.forbiddenContinuation = @('once upon a time', 'lullaby', 'little rabbit', 'little bunny')
        }
    }
} finally {
    $synthesizer.Dispose()
}
$manifest = [ordered]@{
    format = 'voice-lab-multitopic-soak-v1'
    generatedAt = [DateTime]::UtcNow.ToString('o')
    generator = 'Windows System.Speech synthesis; recorded test inputs, not human speech'
    voice = $Voice
    sampleRate = 16000
    channels = 1
    bitsPerSample = 16
    minimumDurationSeconds = 540
    minimumContextRollovers = 5
    capturePacketSamples = 320
    capturePacketIntervalMilliseconds = 20
    turns = $turns
    checks = @(
        'Keep microphone PCM streaming continuously, including silence; do not call finish_input.',
        'Inject turn 2 during audible turn 1; within 1.5 seconds require native yield and flush or native EOS and complete old playback drain, then a correct new-topic answer.',
        'Space later turns across at least five observed context_rolled events, including after long idle silence.',
        'Each new request must produce a response matching its own expectedAny words; score only its response epoch.',
        'No assistant continuation of the abandoned bedtime story after topic change; allow explicit acknowledgement of stopping it.',
        'Record native input transcript for each request to distinguish recognition failures from stale-topic responses.',
        'Measure maximum scheduled playback lead, arrival gaps, output duration, errors, and bounded process memory.',
        'Exercise explicit Interrupt during one later reply and confirm a subsequent different request succeeds.',
        'Do not treat absence of crashes, repeated identical questions, or PCM volume alone as conversation correctness.'
    )
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output $manifestPath
