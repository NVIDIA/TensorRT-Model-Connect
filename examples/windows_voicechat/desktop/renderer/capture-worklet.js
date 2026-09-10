/* Capture continuous microphone audio in 20 ms blocks, ahead of the native
 * 80 ms silence fallback deadline even when device/IPC delivery jitters. */
class VoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.round(sampleRate * 0.02));
    this.offset = 0;
  }

  process(inputs, outputs) {
    // A silent output keeps this node scheduled without microphone feedback.
    for (const output of outputs) for (const channel of output) channel.fill(0);
    const channels = inputs[0];
    if (!channels || !channels[0]) return true;
    for (let i = 0; i < channels[0].length; i += 1) {
      let value = 0;
      for (const channel of channels) value += channel[i] || 0;
      this.block[this.offset++] = value / channels.length;
      if (this.offset === this.block.length) {
        this.port.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(Math.round(sampleRate * 0.02));
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('voice-capture', VoiceCaptureProcessor);
