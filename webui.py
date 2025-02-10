import argparse
import os
import gradio as gr
from loguru import logger
from rag import Rag
import json 

pwd_path = os.path.abspath(os.path.dirname(__file__))

def predict_stream(message, history):
    history_format = []
    for human, assistant in history:
        history_format.append([human, assistant])
    model.history = history_format
    for chunk in model.predict_stream(message,max_length=2048):
        yield chunk

def predict(message, history):
    logger.debug(message)
    response, reference_results,prompt = model.predict(message)
    r = response + "\n\n" + '\n'.join(reference_results)
    logger.debug(r)
    return r

# Define the API function
def api_predict(query):
    response, reference_results,prompt = model.predict(query,max_length=2048)
    reference_results=json.dumps(reference_results,indent=4,ensure_ascii=False)
    return response,reference_results,prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default='')
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files_path", type=str, default="/Users/lee/Documents/work/干预助手/dataset/知识库")
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--share", action='store_true', help="share model")
    args = parser.parse_args()
    logger.info(args)

    model = Rag(
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files_path=args.corpus_files_path,
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )
    # logger.info(f"chatpdf model: {model}")
    # 定义路径
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    user_avatar = os.path.join(pwd_path, "assets/user.png")
    llama_avatar = os.path.join(pwd_path, "assets/llama.png")

    # 检查路径
    if not os.path.exists(user_avatar):
        raise FileNotFoundError(f"User avatar image not found: {user_avatar}")
    if not os.path.exists(llama_avatar):
        raise FileNotFoundError(f"Llama avatar image not found: {llama_avatar}")

    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        # 显式地将两个界面添加到布局中
        with gr.Tabs():
            with gr.TabItem("Chat"):
                # Chat Interface
                chatbot_stream = gr.Chatbot(
                    height=400,
                    avatar_images=(user_avatar, llama_avatar),
                    bubble_full_width=False
                )
                title = " 🎉Chat WebUI🎉 "
                description = ""
                css = """.toast-wrap { display: none !important } """
                examples = ['Can you tell me about the NT?', '介绍下康复师.']
                chat_interface_stream = gr.ChatInterface(
                    predict_stream,
                    textbox=gr.Textbox(lines=4, placeholder="Ask me question", scale=7),
                    title=title,
                    description=description,
                    chatbot=chatbot_stream,
                    css=css,
                    examples=examples,
                    theme='soft',
                )
            
            with gr.TabItem("API"):
                tcss = """
                    .gr-code {
                        white-space: pre-wrap;
                    }
                    """
                # API Interface
                api_interface = gr.Interface(
                    fn=api_predict,
                    inputs=gr.Textbox(lines=2,scale=4,placeholder="Enter your query here"),
                    # outputs=gr.Code(label="output", language="json"),#gr.JSON(height=60,label="Api Result Output"),
                    css=tcss,
                    outputs=[
                            gr.Textbox(lines=10,label="Response"),  # 回应
                            gr.Code(label="Reference Files", scale=20,language="json"),  # 参考文件
                            gr.Textbox(lines=10,label="Prompt"),  # prompt
                            ],
                    title="RAG API",
                    description="API for querying the RAG model"
                )
    # 启动 Gradio 应用
    demo.launch(
        server_name=getattr(args, "server_name", "0.0.0.0"),
        server_port=getattr(args, "server_port", 7860),
        share=getattr(args, "share", False)
    )