#!/usr/bin/env python3
"""
Fish Audio TTS Japanese Voice Agent

Japanese conversation agent using Fish Audio TTS
"""

import logging
from dotenv import load_dotenv

# Load environment variables BEFORE importing plugins
# This is crucial because Fish Audio plugin reads FISHAUDIO_API_KEY at import time
load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero
from livekit.plugins import fishaudio

logger = logging.getLogger("fishaudio-agent")
logger.setLevel(logging.INFO)


class JapaneseAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "あなたは親しみやすい日本語の音声アシスタントです。"
                "自然な日本語で会話をしてください。"
                "返答は簡潔で会話的にしてください。"
                "ユーザーの質問に対して、丁寧かつフレンドリーに答えてください。"
                "日常会話のように自然なトーンで話してください。"
            ),
        )

    async def on_enter(self):
        # Generate initial greeting in Japanese
        self.session.generate_reply(instructions="ユーザーに温かく挨拶してください")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent"""
    logger.info("🚀 Starting Fish Audio Japanese Voice Agent")
    await ctx.connect()

    # Create Japanese agent instance
    agent = JapaneseAgent()

    # Configure the agent session with Fish Audio TTS for Japanese
    logger.info("⚙️  Configuring agent session with Fish Audio TTS")
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(
                model="gpt-4o-mini-transcribe",
                language="ja", use_realtime=True),
        llm=openai.LLM(
                model="gpt-5",
                reasoning_effort="minimal",
                verbosity="low",
                tool_choice="auto",  # Automatically choose tools based on context
            ),
        # Configure Fish Audio TTS for Japanese
        tts=fishaudio.TTS(
            language="ja",  # Japanese language
            temperature=0.7,  # Sampling temperature (0.0 to 1.0)
            top_p=0.7,  # Nucleus sampling parameter
            reference_id="fbea303b64374bffb8843569404b095e"  # Optional: Use a specific Japanese voice
        ),
    )

    # Track conversation metrics and timing
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        """Log total usage summary at the end of the session"""
        summary = usage_collector.get_summary()
        logger.info(f"📊 Session Summary | {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    logger.info("✅ Agent session starting - ready to handle conversations")
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
