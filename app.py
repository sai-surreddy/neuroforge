"""
NeuroForge - Autonomous Software Engineering Agent
AMD Developer Hackathon 2026
"""
import gradio as gr
import requests
import json
import time

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"


def call_qwen(messages, temperature=0.3):
    """Call Qwen model via vLLM."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048
    }
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def plan_task(requirement, tech_stack):
    """Step 1: Create implementation plan."""
    system_msg = "You are an expert software architect. Break down the task into 3-5 clear implementation steps. Return ONLY a numbered list."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Tech Stack: {tech_stack}\n\nRequirement: {requirement}\n\nCreate implementation plan:"}
    ]
    return call_qwen(messages, 0.2)


def generate_code(step, context):
    """Step 2: Generate code for a step."""
    system_msg = "You are a senior software engineer. Write clean, well-commented Python code with error handling."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context: {context}\n\nTask: {step}\n\nWrite the code:"}
    ]
    return call_qwen(messages, 0.2)


def review_code(code, requirements):
    """Step 3: Review code quality."""
    system_msg = "You are a code reviewer. Check for bugs, security issues, and improvements. Return a brief review."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Requirements: {requirements}\n\nCode:\n{code}\n\nReview:"}
    ]
    return call_qwen(messages, 0.3)


def process_request(requirement, tech_stack):
    """Main processing pipeline."""
    results = []

    # Planning
    results.append(("Planning", "Analyzing requirements..."))
    yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    plan = plan_task(requirement, tech_stack)
    results.append(("Plan Ready", f"Generated {len(plan.split(chr(10)))} steps"))
    yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    # Coding
    results.append(("Coding", "Generating code..."))
    yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    steps = [s for s in plan.split(chr(10)) if s.strip() and s[0].isdigit()]
    all_code = []
    for i, step in enumerate(steps[:3], 1):  # Limit to 3 steps for demo
        code = generate_code(step, f"Goal: {requirement}")
        all_code.append(f"# Step {i}: {step}\n{code}")
        results.append((f"Step {i}", "Complete"))
        yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    full_code = "\n\n".join(all_code)

    # Review
    results.append(("Review", "Checking code quality..."))
    yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    review = review_code(full_code, requirement)
    results.append(("Review Complete", "Done"))
    yield "\n".join([f"**{phase}**: {msg}" for phase, msg in results])

    # Final output
    yield f"## Workflow Complete\n\n{chr(10).join([f'**{phase}**: {msg}' for phase, msg in results])}\n\n## Generated Code\n\n```python\n{full_code}\n```\n\n## Review\n\n{review}"


# Gradio UI
with gr.Blocks(title="NeuroForge - AI Software Engineer") as demo:
    gr.Markdown("# 🧠 NeuroForge\n### Autonomous Software Engineering Agent\n**Powered by AMD MI300X + Qwen2.5-Coder**")

    with gr.Row():
        with gr.Column(scale=1):
            req_input = gr.Textbox(label="What do you want to build?", lines=5, placeholder="Describe your project...")
            stack_input = gr.Dropdown(label="Tech Stack", choices=["Python/FastAPI", "Python/Flask", "Node.js/Express"], value="Python/FastAPI")
            btn = gr.Button("Generate Software", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Results")

    btn.click(fn=process_request, inputs=[req_input, stack_input], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
