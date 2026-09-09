from nicegui import ui
from backend import Backend
from uart_backend import UartBackend
import sys

CLASS_NAMES = ["Jogging", "Stationary", "Stairs", "Walking"]

default_serial_port = "/dev/ttyACM0"
default_server_port = 8080
serial_port = sys.argv[1] if len(sys.argv) > 1 else default_serial_port
web_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

backend: Backend = UartBackend(serial_port)

# Page styling & container
ui.colors(primary="#2563eb", secondary="#64748b", accent="#0ea5e9")

with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-4"):
    # Header
    with ui.row().classes("w-full items-center justify-between pb-2 border-b"):
        ui.label("HAR Dashboard").classes("text-2xl font-bold tracking-tight text-slate-800")
        with ui.row().classes("items-center gap-2"):
            ui.label("Status:").classes("text-sm text-gray-500 font-medium")
            ui.label().bind_text_from(
                backend, "connected",
                backward=lambda c: "Connected" if c else "Disconnected"
            ).classes("text-sm font-semibold uppercase px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700")

    # 2-Column Responsive Layout
    with ui.grid().classes("grid-cols-1 lg:grid-cols-12 gap-4 w-full"):
        
        # --- LEFT COLUMN (Sensors & Logs) ---
        with ui.column().classes("lg:col-span-7 gap-4"):
            # Sensor Signal Card
            with ui.card().classes("w-full shadow-sm border border-slate-100"):
                ui.label("Live Sensor Signal").classes("text-base font-semibold text-slate-700 mb-2")
                plot = ui.line_plot(n=2, limit=150, figsize=(7.5, 3.2), update_every=1) \
                    .with_legend(["Dyn (g)", "Gyro (dps)"], loc="upper left")

            # Raw Log & Command Card
            with ui.card().classes("w-full shadow-sm border border-slate-100 flex-grow"):
                ui.label("Terminal & Raw Output").classes("text-base font-semibold text-slate-700 mb-1")
                log = ui.log(max_lines=200).classes("w-full h-44 font-mono text-xs bg-slate-900 text-emerald-400 p-2 rounded")
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    command = ui.input(placeholder="Enter command (e.g. start)").props("dense outlined").classes("flex-grow")
                    command.value = "start"
                    ui.button("Send", on_click=lambda: backend.send(command.value)).props("unelevated color=primary")

        # --- RIGHT COLUMN (Controls, Inference & Feedback) ---
        with ui.column().classes("lg:col-span-5 gap-4"):
            # Device Controls
            with ui.card().classes("w-full shadow-sm border border-slate-100"):
                ui.label("Device Controls").classes("text-base font-semibold text-slate-700 mb-2")
                with ui.row().classes("flex-wrap gap-2"):
                    ui.button("Connect", on_click=backend.connect).props("unelevated color=primary")
                    ui.button("Disconnect", on_click=backend.disconnect).props("outline color=secondary")
                    ui.button("Start", on_click=lambda: backend.send("start")).props("unelevated color=positive")
                    ui.button("Stop", on_click=lambda: backend.send("stop")).props("unelevated color=negative")
                    ui.button("LED", on_click=lambda: backend.send("led")).props("outline color=primary")

            # Classification Card
            with ui.card().classes("w-full shadow-sm border border-slate-100"):
                ui.label("Inference").classes("text-base font-semibold text-slate-700")
                predicted_label = ui.label("Predicted: -").classes("text-lg font-bold text-blue-600 my-1")
                
                prob_bars = []
                with ui.column().classes("w-full gap-2 mt-1"):
                    for name in CLASS_NAMES:
                        with ui.row().classes("items-center w-full gap-2"):
                            ui.label(name).classes("w-20 text-sm font-medium text-slate-600")
                            bar = ui.linear_progress(value=0.0, show_value=False).classes("flex-grow rounded-full")
                            pct = ui.label("0%").classes("w-10 text-xs font-mono text-right text-slate-500")
                            prob_bars.append((bar, pct))

            # Training Feedback Card
            with ui.card().classes("w-full shadow-sm border border-slate-100"):
                ui.label("Corrective SGD Training").classes("text-base font-semibold text-slate-700")
                ui.label(
                    "Tap the label matching your activity. The board will run one corrective SGD step on the next completed window."
                ).classes("text-xs text-slate-500 mb-2 leading-relaxed")
                
                with ui.grid().classes("grid-cols-2 gap-2 w-full"):
                    for idx, name in enumerate(CLASS_NAMES):
                        ui.button(f"This is: {name}", on_click=lambda i=idx: backend.send(f"l{i}")).props("outline dense")

                ui.button("Finish Round (Send Weights to FL Server)", on_click=lambda: backend.send("done")).props("unelevated color=positive").classes("mt-3 w-full font-bold")

                ui.button("Force Inference", on_click=lambda: backend.send("infer")).props("flat dense color=primary").classes("mt-1 w-full")
                training_status = ui.label("").classes("text-xs text-slate-600 italic mt-1")

# --- Wiring the backend callbacks (Unchanged) ---
_t = 0.0


def _on_prediction(cls: int, probs, dyn: float, gyro: float) -> None:
    global _t
    _t += 0.1
    plot.push([_t], [[dyn], [gyro]])

    name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"Class {cls}"
    confidence = probs[cls] if 0 <= cls < len(probs) else 0.0
    predicted_label.set_text(f"Predicted: {name} ({confidence * 100:.1f}%)")

    for (bar, pct), p in zip(prob_bars, probs):
        bar.set_value(p)
        pct.set_text(f"{p * 100:.0f}%")


def _on_training(cls: int, loss: float, dyn: float, gyro: float) -> None:
    name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"Class {cls}"
    training_status.set_text(f"Trained on '{name}' — loss {loss:.4f}")
    log.push(f"[train] label={cls} ({name}) loss={loss:.4f} dyn={dyn:.2f}g gyro={gyro:.1f}dps")


backend.on_prediction = _on_prediction
backend.on_training = _on_training
backend.on_message = log.push

ui.run(
    title=f"HAR Dashboard ({serial_port})",
    port=web_port,
    reload=False,  # Prevents port-rebinding issues on restart
    # show=False
)