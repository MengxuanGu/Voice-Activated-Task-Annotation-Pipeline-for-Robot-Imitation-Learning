import threading
import keyboard
from pathlib import Path
from datetime import datetime

from sound_recorder import SoundRecorder
from tiago_dual import TiagoTeleopSession
from robot_recorder_tiago import HDF5Recorder
from whisper_transcriber import WhisperTranscriber
from api_keyword_extractor import GPTTextProcessor
from playback_tiago import PlaybackTiago


def find_latest_h5(folder: Path) -> str:
    cands = sorted(folder.glob("*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else ""


def main():
    # Path
    proj_root = Path(__file__).parent
    traj_dir = proj_root / "trajectory_storage"
    audio_dir = proj_root / "sound_storage"
    traj_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # The root directory of the Tiago scene's XML (containing the included files)
    xml_root_dir = proj_root / "models" / "pal_tiago_dual"
    xml_main = xml_root_dir / "tiago_scene.xml"
    if not xml_main.exists():
        print(f"[WARN] tiago_scene.xml not found at: {xml_main}")
        print("       Please set xml_root_dir to the folder that contains tiago_scene.xml and its included files.")
    
    # Standardized timestamp prefix (for easy archiving)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize audio recorder
    sound_recorder = SoundRecorder(folder=str(audio_dir))

    # Initialize the TIAGO track recorder (non-robosuite environment)
    recorder = HDF5Recorder(
        out_dir=str(traj_dir),
        env_name="CustomMuJoCo_TIAGO",
        env_info={"controller": "mink_ik", "note": "keyboard teleop with mocap targets"},
        state_after_action=False,
    )
    session = TiagoTeleopSession(recorder=recorder)

    # ====== Waiting to begin ======
    print("Press 's' to start recording sound + TIAGO trajectory")
    keyboard.wait('s')
    print("[START] 's' detected. Start synchronized recording!")

    # 1) Start recording (thread)
    sound_recorder.is_recording = True
    audio_thread = threading.Thread(target=sound_recorder.record_audio, daemon=True)
    audio_thread.start()

    # 2) Start TIAGO screen + trajectory (thread)
    stop_event = threading.Event()
    session.stop_event = stop_event
    robot_thread = threading.Thread(target=session.run, daemon=True)
    robot_thread.start()

    # ====== Waiting to end ======
    print("Press 'e' to stop recording")
    keyboard.wait('e')
    print("[STOP] 'e' detected. Stopping...")

    # Stop both threads
    sound_recorder.is_recording = False
    stop_event.set()

    # Waiting for the end
    audio_thread.join()
    robot_thread.join()

    # Find the latest audio and track files
    latest_h5 = find_latest_h5(traj_dir)
    print(f"[OK] Trajectory saved under: {latest_h5 or traj_dir}")
    print(f"[OK] Audio folder: {audio_dir}")

    # ====== Whisper transcribe ======
    print("\nStart transcribing ...")
    transcriber = WhisperTranscriber(model_size="base")
    audio_path = transcriber.find_latest_wav(folder=str(audio_dir))
    result = transcriber.transcribe(audio_path)
    result_with_timestamps = transcriber.word_timestamps(result)

    # ====== Extracting action units (based on transcribed text) ======
    print("\nExtract motion units ...")
    extractor = GPTTextProcessor()
    processed_result = extractor.process_text(result_with_timestamps)
    print("\nMotion units are as follows")
    print(processed_result)

    # ====== Replay (Press p to start replaying the latest demo)======
    if latest_h5:
        print("\nPress 'p' to playback the latest TIAGO trajectory (or any other key to skip).")
        try:
            keyboard.wait('p')
            print("[PLAY] 'p' detected. Launching playback...")
            player = PlaybackTiago(
                h5_path=latest_h5,
                demo=None,        # The first demo will be replayed by default.
                mode="action",    # Or "state"
                speed=1.0,
                control_hz=200.0,
                loop=False,
                xml_root_dir=str(xml_root_dir),
            )
            player.run()
        except KeyboardInterrupt:
            pass
    else:
        print("[WARN] No .hdf5 file found in trajectory_storage. Skip playback.")


if __name__ == "__main__":
    main()
