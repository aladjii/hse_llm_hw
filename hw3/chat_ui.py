import html
import json

import ipywidgets as widgets
from IPython.display import display

from smart_home import SYSTEM_PROMPT

CSS = """<style>
.fc-chat {font-family:-apple-system,Inter,sans-serif;line-height:1.5}
.fc-bubble {padding:10px 12px;margin:8px;border-radius:14px;white-space:pre-wrap}
.fc-user {background:#153b68;color:white;margin-left:20%}
.fc-assistant {background:#eee;color:#222;margin-right:15%}
.fc-tool {border:1px solid #aaa;border-radius:10px;padding:8px 12px;margin:4px 8px;font-family:monospace;white-space:pre-wrap}
.fc-error {border-color:#d44}
</style>"""


def render_messages(messages):
    """HTML диалога: реплики user/assistant — пузыри, вызовы функций и их результаты — карточки."""
    pieces = []
    for message in messages:
        if message["role"] == "user":
            pieces.append(f'<div class="fc-bubble fc-user">{html.escape(message["content"])}</div>')
        elif message["role"] == "assistant":
            if message.get("content"):
                pieces.append(f'<div class="fc-bubble fc-assistant">{html.escape(message["content"])}</div>')
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = json.dumps(function["arguments"], ensure_ascii=False)
                pieces.append(f'<div class="fc-tool">🔧 {html.escape(function["name"])}({html.escape(arguments)})</div>')
        elif message["role"] == "tool":
            error_class = " fc-error" if '"error"' in message["content"] else ""
            pieces.append(f'<div class="fc-tool{error_class}">↩ {html.escape(message["content"])}</div>')
    return f'{CSS}<div class="fc-chat">{"".join(pieces)}</div>'


def render_home(home):
    state = home.snapshot()
    rooms = "".join(
        f'<p><b>{room}</b><br>{"💡" if data["light"] == "on" else "⚫"} {data["light"]} · 🌡 {data["temperature"]}°C</p>'
        for room, data in state["rooms"].items()
    )
    shopping = html.escape(json.dumps(state["shopping_list"], ensure_ascii=False))
    timers = html.escape(json.dumps(state["timers"], ensure_ascii=False))
    return f"<h3>🏠 Состояние дома</h3>{rooms}<p>🛒 {shopping}</p><p>⏱ {timers}</p>"


def launch_chat(model, tokenizer, home, run_agent):
    """Чат с моделью: слева диалог, справа состояние дома. run_agent дописывает ответы в messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript = widgets.HTML(layout=widgets.Layout(height="440px", overflow="auto"))
    panel = widgets.HTML()
    entry = widgets.Text(placeholder="Ask your smart home assistant...", layout=widgets.Layout(width="80%"))
    send = widgets.Button(description="Отправить", button_style="primary")
    reset = widgets.Button(description="Сбросить дом")

    def render():
        transcript.value = render_messages(messages)
        panel.value = render_home(home)

    def respond(_):
        text = entry.value.strip()
        if not text:
            return
        entry.value = ""
        messages.append({"role": "user", "content": text})
        render()
        try:
            run_agent(model, tokenizer, home, messages)
        except Exception as error:
            # Исключение внутри callback виджета иначе не видно в ноутбуке.
            messages.append({"role": "assistant", "content": f"Ошибка: {error}"})
        render()

    def clear(_):
        home.reset()
        del messages[1:]
        render()

    send.on_click(respond)
    entry.on_submit(respond)
    reset.on_click(clear)
    render()
    display(widgets.HBox([
        widgets.VBox([transcript, widgets.HBox([entry, send])], layout=widgets.Layout(width="65%")),
        widgets.VBox([panel, reset], layout=widgets.Layout(width="35%")),
    ]))
