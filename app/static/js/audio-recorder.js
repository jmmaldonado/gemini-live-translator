/**
 * Audio Recorder Worklet
 */

export async function startAudioRecorderWorklet(audioRecorderHandler, deviceId, useTabAudio = false) {
  const audioRecorderContext = new AudioContext({ sampleRate: 16000 });
  const workletURL = new URL("./pcm-recorder-processor.js", import.meta.url);
  await audioRecorderContext.audioWorklet.addModule(workletURL);

  let stream;
  if (useTabAudio) {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: true
    });
    const audioTracks = stream.getAudioTracks();
    if (audioTracks.length === 0) {
      stream.getTracks().forEach(track => track.stop());
      throw new Error("No tab audio stream available. Did you check 'Share audio'?");
    }
  } else {
    const constraints = { audio: { channelCount: 1 } };
    if (deviceId) constraints.audio.deviceId = { exact: deviceId };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  }

  const audioStream = useTabAudio ? new MediaStream([stream.getAudioTracks()[0]]) : stream;
  const source = audioRecorderContext.createMediaStreamSource(audioStream);

  const audioRecorderNode = new AudioWorkletNode(
    audioRecorderContext,
    "pcm-recorder-processor"
  );

  source.connect(audioRecorderNode);
  audioRecorderNode.port.onmessage = (event) => {
    const pcmData = convertFloat32ToPCM(event.data);
    audioRecorderHandler(pcmData);
  };
  return [audioRecorderNode, audioRecorderContext, stream];
}

function convertFloat32ToPCM(inputData) {
  const pcm16 = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    pcm16[i] = inputData[i] * 0x7fff;
  }
  return pcm16.buffer;
}
