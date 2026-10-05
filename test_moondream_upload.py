import asyncio

from ml_pipeline.model_manager import ModelManager
from ml_pipeline.vision.qwen_vl_analyzer import QwenVLAnalyzer


async def main():
    model_manager = ModelManager()
    analyzer = QwenVLAnalyzer(model_manager)

    await analyzer.initialize()

    image_path = r".\uploads\016d73b83afd470e9f232077da211364.png"

    print("\nTesting exact dashboard upload...")
    print("Image:", image_path)

    result = await analyzer.analyze_skin(image_path)

    print("\n========== RESULT ==========")
    print("Status:", result.get("status"))
    print("Model:", result.get("model"))
    print("Description:")
    print(result.get("description"))
    print("============================")


if __name__ == "__main__":
    asyncio.run(main())