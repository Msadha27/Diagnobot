import asyncio

from ml_pipeline.model_manager import ModelManager
from ml_pipeline.vision.qwen_vl_analyzer import QwenVLAnalyzer


async def main():
    print("1. Creating ModelManager...")
    model_manager = ModelManager()

    print("2. Creating QwenVLAnalyzer...")
    analyzer = QwenVLAnalyzer(model_manager)

    print("3. Initializing Moondream...")
    await analyzer.initialize()

    print("4. Running image analysis...")

    result = await analyzer.analyze_general(
        r".\uploads\54c87c11fcc745b7b0308af96c6b38b0.jpg"
    )

    print("\n========== RESULT ==========")
    print("Status:", result.get("status"))
    print("Model:", result.get("model"))
    print("Description:")
    print(result.get("description"))
    print("============================")


if __name__ == "__main__":
    asyncio.run(main())