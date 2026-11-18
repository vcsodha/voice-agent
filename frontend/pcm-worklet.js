// Runs inside the browser's audio thread. Takes the mic audio, resamples it
// to 16kHz and sends it to the page in 30ms pieces (960 bytes of 16-bit audio)
// which is the frame size the server's voice activity detector wants.
class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetSampleRate;
    this.inputRate = options.processorOptions.inputSampleRate;
    this.ratio = this.inputRate / this.targetRate;
    this.frameSamples = Math.round(this.targetRate * 0.03); // 30ms
    this.buffer = [];
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    for (let i = 0; i < input.length; i += this.ratio) {
      const idx = Math.floor(i);
      this.buffer.push(input[idx] || 0);
    }

    while (this.buffer.length >= this.frameSamples) {
      const frame = this.buffer.splice(0, this.frameSamples);
      const pcm16 = new Int16Array(frame.length);
      for (let j = 0; j < frame.length; j++) {
        const s = Math.max(-1, Math.min(1, frame[j]));
        pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-downsampler', PCMDownsampler);
