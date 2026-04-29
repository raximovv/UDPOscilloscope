import socket
import struct
import threading
import time
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = "127.0.0.1"
PORT = 5005
SEND_RATE_HZ = 200
BUFFER_SIZE = 512
LOSS_RATE = 0.10

PACKET_FORMAT = "!Hfd"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

recv_buffer = collections.deque(maxlen=BUFFER_SIZE)
recv_lock = threading.Lock()

stats = {
    "sent": 0,
    "received": 0,
    "lost": 0,
    "loss_pct": 0.0,
    "last_seq": None,
}

stats_lock = threading.Lock()

loss_enabled = threading.Event()
noise_enabled = threading.Event()
running = True


def sender_thread():
    global running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    start = time.time()
    delay = 1.0 / SEND_RATE_HZ

    while running:
        t = time.time() - start
        value = np.sin(2 * np.pi * 2.0 * t)

        if noise_enabled.is_set():
            value += np.random.normal(0, 0.25)

        packet = struct.pack(PACKET_FORMAT, seq, t, value)

        should_drop = loss_enabled.is_set() and np.random.random() < LOSS_RATE

        if not should_drop:
            sock.sendto(packet, (HOST, PORT))

        with stats_lock:
            stats["sent"] += 1

        seq = (seq + 1) % 65536
        time.sleep(delay)

    sock.close()


def receive_packets(sock):
    global running

    sock.settimeout(0.1)

    while running:
        try:
            data, _ = sock.recvfrom(1024)

            if len(data) < PACKET_SIZE:
                continue

            seq, t, value = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])

            with stats_lock:
                last_seq = stats["last_seq"]

                if last_seq is not None:
                    expected = (last_seq + 1) % 65536
                    if seq != expected:
                        gap = (seq - expected) % 65536
                        if 0 < gap < 1000:
                            stats["lost"] += gap

                stats["last_seq"] = seq
                stats["received"] += 1

                total = stats["received"] + stats["lost"]
                if total > 0:
                    stats["loss_pct"] = 100.0 * stats["lost"] / total

            with recv_lock:
                recv_buffer.append(value)

        except socket.timeout:
            continue

        except Exception:
            continue


def run_receiver():
    global running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((HOST, PORT))
    except OSError as e:
        print(f"Cannot bind to {HOST}:{PORT}: {e}")
        sys.exit(1)

    thread = threading.Thread(target=receive_packets, args=(sock,), daemon=True)
    thread.start()

    fig, (ax_wave, ax_info) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    fig.suptitle("UDP Oscilloscope")

    ax_wave.set_xlim(0, BUFFER_SIZE)
    ax_wave.set_ylim(-2, 2)
    ax_wave.set_title("Live UDP Sensor Stream")
    ax_wave.set_xlabel("Samples")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True)

    line, = ax_wave.plot([], [], linewidth=1.5)

    status_text = ax_wave.text(
        0.02,
        0.92,
        "",
        transform=ax_wave.transAxes,
        fontsize=10,
        family="monospace",
    )

    loss_text = ax_wave.text(
        0.72,
        0.92,
        "",
        transform=ax_wave.transAxes,
        fontsize=10,
        family="monospace",
    )

    ax_info.axis("off")

    info_text = ax_info.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        transform=ax_info.transAxes,
    )

    def update(_):
        with recv_lock:
            values = list(recv_buffer)

        if not values:
            return line, status_text, loss_text, info_text

        x = np.arange(len(values))
        line.set_data(x, values)

        with stats_lock:
            sent = stats["sent"]
            received = stats["received"]
            lost = stats["lost"]
            loss_pct = stats["loss_pct"]

        mode = []

        if loss_enabled.is_set():
            mode.append("LOSS ON")

        if noise_enabled.is_set():
            mode.append("NOISE ON")

        mode_text = " | ".join(mode) if mode else "CLEAN"

        status_text.set_text(f"Rate: {SEND_RATE_HZ} Hz | Mode: {mode_text}")
        loss_text.set_text(f"Loss: {loss_pct:.2f}%")

        info_text.set_text(
            f"Sent: {sent:,}   Received: {received:,}   "
            f"Lost: {lost:,}   Packet Loss: {loss_pct:.2f}%"
        )

        return line, status_text, loss_text, info_text

    def on_key(event):
        global running

        if event.key == "l":
            if loss_enabled.is_set():
                loss_enabled.clear()
                print("Packet loss OFF")
            else:
                loss_enabled.set()
                print("Packet loss ON")

        elif event.key == "n":
            if noise_enabled.is_set():
                noise_enabled.clear()
                print("Noise OFF")
            else:
                noise_enabled.set()
                print("Noise ON")

        elif event.key == "q":
            running = False
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    print("UDP Oscilloscope running")
    print("Press l = toggle packet loss")
    print("Press n = toggle noise")
    print("Press q = quit")

    plt.tight_layout()
    plt.show()

    running = False
    sock.close()


if __name__ == "__main__":
    sender = threading.Thread(target=sender_thread, daemon=True)
    sender.start()

    time.sleep(0.2)

    try:
        run_receiver()
    except KeyboardInterrupt:
        running = False
        print("Stopped")