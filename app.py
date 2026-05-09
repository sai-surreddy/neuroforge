"""
NeuroForge - Autonomous Software Engineering Agent
AMD Developer Hackathon 2026
"""
import gradio as gr
import requests
import json
import tempfile
import os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"


def call_qwen(messages, temperature=0.3):
    """Call Qwen model via vLLM."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False
    }
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def call_qwen_stream(messages, temperature=0.3):
    """Call Qwen model via vLLM with streaming."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": True
    }
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"Error: {str(e)}"


def plan_task(requirement, tech_stack):
    """Step 1: Create implementation plan."""
    system_msg = "You are an expert software architect. Break down the task into 3-5 clear implementation steps. Return ONLY a numbered list."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Tech Stack: {tech_stack}\n\nRequirement: {requirement}\n\nCreate implementation plan:"}
    ]
    return call_qwen(messages, 0.2)


def generate_code_stream(step, context):
    """Step 2: Generate code for a step with streaming."""
    system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context: {context}\n\nTask: {step}\n\nWrite the code:"}
    ]
    return call_qwen_stream(messages, 0.2)


def review_code(code, requirements):
    """Step 3: Review code quality."""
    system_msg = "You are a code reviewer. Check for bugs, security issues, and improvements. Return a brief review."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Requirements: {requirements}\n\nCode:\n{code}\n\nReview:"}
    ]
    return call_qwen(messages, 0.3)


def save_code_to_file(code):
    """Save generated code to a temp file and return path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="neuroforge_"
    )
    tmp.write(code)
    tmp.close()
    return tmp.name


def process_request(requirement, tech_stack):
    """Main processing pipeline with real-time streaming updates."""
    log_lines = []
    all_code_blocks = []

    def log(icon, phase, msg):
        log_lines.append(f"{icon} **{phase}**: {msg}")
        return "\n\n".join(log_lines)

    # Step 1: Plan
    status = log("🔍", "Planning", "Analyzing requirements...")
    yield status, "*Generating plan...*", None

    plan = plan_task(requirement, tech_stack)
    steps = [s for s in plan.split("\n") if s.strip() and s[0].isdigit()]

    status = log("📋", "Plan", f"{len(steps)} steps identified")
    yield status, f"**Plan:**\n```\n{plan}\n```", None

    # Step 2: Code generation (streamed per step)
    for i, step in enumerate(steps[:3], 1):
        status = log("⚙️", f"Step {i}/{min(len(steps), 3)}", step.strip())
        yield status, f"**Plan:**\n```\n{plan}\n```\n\n*Generating step {i}...*", None

        system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context: Goal: {requirement}\n\nTask: {step}\n\nWrite the code:"}
        ]

        streamed = f"# Step {i}: {step.strip()}\n"
        for token in call_qwen_stream(messages, 0.2):
            streamed += token
            current_code = "\n\n".join(all_code_blocks) + ("\n\n" if all_code_blocks else "") + streamed
            yield status, f"**Plan:**\n```\n{plan}\n```\n\n**Generated Code:**\n```python\n{current_code}\n```", None

        all_code_blocks.append(streamed)
        status = log("✅", f"Step {i}", "Complete")

    generated_code = "\n\n".join(all_code_blocks)

    # Step 3: Review
    status = log("🔎", "Review", "Checking code quality...")
    yield status, f"**Generated Code:**\n```python\n{generated_code}\n```", None

    review = review_code(generated_code, requirement)
    status = log("🎉", "Done", "All steps complete!")

    # Save file for download
    tmp_path = save_code_to_file(generated_code)

    final_md = f"**Generated Code:**\n```python\n{generated_code}\n```\n\n---\n\n**Code Review:**\n\n{review}"
    yield status, final_md, tmp_path


# Gradio UI
with gr.Blocks(title="NeuroForge - AI Software Engineer") as demo:

    gr.Markdown(
        """# 🧠 NeuroForge
### Autonomous Software Engineering Agent
**Powered by AMD MI300X + Qwen2.5-Coder-14B**"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            req_input = gr.Textbox(
                label="What do you want to build?",
                lines=6,
                placeholder="e.g. A REST API for user authentication with JWT tokens...",
            )
            stack_input = gr.Dropdown(
                label="Tech Stack",
                choices=[
                    "Python/FastAPI",
                    "Python/Flask",
                    "Python/Django",
                    "Node.js/Express",
                    "Python/Streamlit",
                ],
                value="Python/FastAPI",
            )
            btn = gr.Button("⚡ Generate Software", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### 🤖 Agent Log")
            status_out = gr.Markdown(value="*Waiting for task...*")

        with gr.Column(scale=2):
            gr.Markdown("### 📄 Generated Code")
            code_out = gr.Markdown(value="*Code will appear here as it is generated...*")
            download_btn = gr.File(label="⬇️ Download Generated Code (.py)", visible=False)

    def run(req, stack):
        for status, code, tmp in process_request(req, stack):
            if tmp:
                yield status, code, gr.update(value=tmp, visible=True)
            else:
                yield status, code, gr.update(visible=False)

    btn.click(
        fn=run,
        inputs=[req_input, stack_input],
        outputs=[status_out, code_out, download_btn],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
