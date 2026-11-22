# Render SSML clips to 16 kHz mono WAV with the Windows built-in voices.
# Used by make_tts_corpus.py:  powershell -File tts_render.ps1 <manifest.json> <out_dir>
param([string]$Manifest, [string]$OutDir)

Add-Type -AssemblyName System.Speech
$clips = Get-Content -Raw -Encoding UTF8 $Manifest | ConvertFrom-Json
New-Item -ItemType Directory -Force $OutDir | Out-Null
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($clip in $clips) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice($clip.voice)
    $synth.SetOutputToWaveFile((Join-Path $OutDir ($clip.id + ".wav")), $fmt)
    $synth.SpeakSsml($clip.ssml)
    $synth.Dispose()
}
Write-Output ("rendered {0} clips" -f @($clips).Count)
