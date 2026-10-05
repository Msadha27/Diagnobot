import asyncio

from ml_pipeline.model_manager import ModelManager
from ml_pipeline.nlp.report_generator import ReportGenerator


async def main():
    print("1. Creating ModelManager...")
    model_manager = ModelManager()

    print("2. Creating ReportGenerator...")
    generator = ReportGenerator(model_manager)

    print("3. Initializing Llama 3.2...")
    await generator.initialize()

    print("4. Generating clinical reasoning...")

    result = await generator.generate_report(
        """
Clinical summary: Patient reports mild redness and itching on the skin.

Visual analysis description:
The image shows a localized area of skin redness.

Detected condition hint:
Possible dermatological irritation.

Risk flags:
None reported.

Generate a cautious medical decision-support summary.
Do not provide a definitive diagnosis.
""",
        max_length=96,
    )

    print("\n========== LLAMA RESULT ==========")
    print(result)
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main())