import asyncio
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import boto3
import numpy as np
import re

class ClaudeChat:
    def __init__(self, region_name='us-west-2'):
        # 初始化 bedrock 客户端和对话历史
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region_name)
        self.conversation_history = ""

    def chat_with_claude(self, prompt):
        # 构建请求体
        # 直接调用大模型 0001
        # native_request = {
        #     "anthropic_version": "bedrock-2023-05-31",
        #     "max_tokens": 2048,
        #     "temperature": 0.5,
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [{"type": "text", "text": prompt}],
        #         }
        #     ],
        # }
        # request = json.dumps(native_request)
        # modelID1='anthropic.claude-3-5-sonnet-20241022-v2:0'
        # modelID2='anthropic.claude-3-haiku-20240307-v1:0'
        try:
            # 调用模型
            # 直接调用大模型
            # response = self.bedrock.invoke_model(
            #     modelId=modelID2,
            #     body=request
            # )

            # 解析响应
            # response_body = json.loads(response["body"].read())
            # return response_body["content"][0]["text"]
            response = self.bedrock_agent_runtime.invoke_flow(
                flowIdentifier='AAAAAAAAA',
                # Amazon Nova Lite
                #flowAliasIdentifier='XXXXXXX',

                # Claude-Haiku 
                flowAliasIdentifier='YYYYYYY',
                inputs=[
                    {
                        "content": {
                            "document": prompt
                        },
                        "nodeName": "FlowInputNode",
                        "nodeOutputName": "document"
                    }
                ]
            )
            event_stream = response["responseStream"]
            for event in event_stream:
                if "flowOutputEvent" in event:
                    response_body = event
            return response_body["flowOutputEvent"]["content"]["document"]
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def chat(self, user_input):
        """进行单轮对话，并保持对话历史"""
        # 添加用户输入到对话历史
        #self.conversation_history += f"\n\nHuman: {user_input}\n\nAssistant:"
        self.conversation_history = user_input
        print(self.conversation_history)
        # 获取响应
        response = self.chat_with_claude(self.conversation_history)

        # 更新对话历史
        if response:
            self.conversation_history += response

        return response

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = ""

    def get_history(self):
        """获取完整对话历史"""
        return self.conversation_history


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, claude_chat_instance):
        super().__init__(output_stream)
        self.claude_chat_instance = claude_chat_instance

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    transcript = alt.transcript
                    print(f"You said: {transcript}")

                    response = self.claude_chat_instance.chat(transcript)
                    if response:
                        print(f"ChatBot: {response}")


# 获取音频流，从本地麦克风采集音频
def get_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    silence_chunk = (np.zeros(1024, dtype=np.int16).tobytes())  # 创建静音帧
    try:
        # while True:
        #     try:
        #         audio_chunk = stream.read(1024, exception_on_overflow=False)
        #         yield audio_chunk
        #     except OSError as e:
        #         print(f"Error occurred while capturing audio: {e}")
        while True:
            audio_chunk = stream.read(1024, exception_on_overflow=False)
            if audio_chunk:
                yield audio_chunk
            else:
                yield silence_chunk
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# 实时语音转录并与 Claude 交互
async def transcribe_and_chat():
    client = TranscribeStreamingClient(region="us-west-2")
    stream = await client.start_stream_transcription(
        language_code="zh-CN",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    claude_chat = ClaudeChat(region_name="us-west-2")
    handler = MyEventHandler(stream.output_stream, claude_chat)

    async def write_chunks():
        for audio_chunk in get_audio_stream():
            await stream.input_stream.send_audio_event(audio_chunk=audio_chunk)
        await stream.input_stream.end_stream()

    await asyncio.gather(write_chunks(), handler.handle_events())


# 主函数
if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(transcribe_and_chat())
    finally:
        loop.close()
