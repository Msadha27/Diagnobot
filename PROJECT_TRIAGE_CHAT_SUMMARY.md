# DiagnoBot Triage Project Summary

## Current Project Framing

DiagnoBot should be presented as a medical triage decision-support prototype, not as a diagnostic tool.

The safest conference claim is:

> DiagnoBot is a multimodal AI triage assistant for skin and wound-related concerns, integrating webcam/image analysis with symptom and report extraction to support urgency assessment and specialist referral.

This means the system does not claim to diagnose disease. It helps decide whether the case looks routine, needs review soon, or needs emergency attention.

## What The Project Currently Does

- Captures or uploads skin/wound images.
- Runs a skin image classifier for probability-based visual hints.
- Uses a vision-language model to describe visible findings.
- Extracts symptoms, report values, abnormal labs, and risk flags from text/PDF inputs.
- Combines visual findings, symptoms, temperature, classifier confidence, and report flags into a triage recommendation.
- Shows recommended urgency, specialist, reasons, red flags, next steps, and safety notes.
- Stores analysis history and generated reports.

## Architecture

```text
Frontend Dashboard
  -> Upload
  -> Webcam
  -> Emergency triage
  -> Doctor second-opinion workspace

FastAPI Backend
  -> Unified upload route
  -> Dermatology route
  -> NLP extraction route
  -> Emergency triage route
  -> Doctor workspace route

Model / Rule Layers
  -> MobileNetV3Small classifier
  -> Moondream2 vision-language observation layer
  -> Rule-based clinical NLP extractor
  -> Gemma-2 reasoning summary
  -> Explainable triage rules

Database
  -> Uploads
  -> Analysis records
  -> Generated reports
```

## Model Roles And How To Explain Them

### MobileNetV3Small

Purpose: skin image classification.

Why it is used:

MobileNetV3Small is an image classification backbone. It extracts visual patterns such as color, texture, surface appearance, and lesion-like structure, then produces class-level probability scores.

Conference wording:

> MobileNetV3Small is used as the image classification backbone because it is well suited for extracting compact visual features from skin images and producing class-level probability scores.

### Moondream2 Vision-Language Model

Purpose: visible finding description.

Why it is used:

A classifier gives only labels and confidence scores. A vision-language model can describe visible findings such as redness, swelling, irregular border, discharge, bleeding, or lesion-like appearance.

Conference wording:

> Moondream2 is used as a vision-language observation layer. It converts the captured image into a descriptive visual summary, focusing on observable signs rather than making a final diagnosis.

### Rule-Based Clinical NLP Extractor

Purpose: symptom and report extraction.

Why it is used:

Triage needs traceable findings. A rule-based extractor can clearly show which symptoms, lab values, abnormal ranges, and risk flags were detected.

Conference wording:

> A rule-based clinical extraction layer is used for symptoms and lab reports because triage requires transparent, traceable findings rather than opaque text classification alone.

### Gemma-2

Purpose: clinical-style summary generation.

Why it is used:

After extracting visual and text findings, Gemma-2 converts them into cautious decision-support language. It should not be presented as the final decision-maker.

Conference wording:

> Gemma-2 is used as the reasoning and summarization layer. It receives structured findings from visual and text analysis and generates a cautious decision-support summary without claiming a definitive diagnosis.

## Important Test Output Interpretation

In the tested skin lesion image, the classifier returned:

```json
"label": "Uncertain",
"confidence": 0.2357,
"top_match": "acne"
```

This means the classifier did not confidently diagnose the image. It only found acne as the closest visual match, with low confidence.

The vision-language model described:

- small red bump or lesion
- raised/irregular shape
- redness
- slight swelling
- possible blood or discharge

The triage layer then returned:

```json
"urgency": "soon",
"recommended_specialist": "Dermatologist"
```

This is the correct triage behavior. The system did not say "this is acne." It said the image is uncertain, but visible warning signs and pain justify dermatologist review soon.

## Why The Output Said "Lesion" But Classification Did Not

"Lesion" is a visible finding, not a disease class in the classifier.

The classifier classes include categories such as acne, impetigo, eczema, candidiasis, molluscum, and nevus. It cannot output "lesion" as the class unless "lesion" is trained as a class.

So the correct interpretation is:

> The system detected lesion-like appearance visually through the VLM description, while the classifier produced disease-category probability hints.

This is medically safer because lesion is an observation, not a diagnosis.

## What Was Added To Improve The Project

The project now includes an explainable triage assessment layer.

New file:

```text
ml_pipeline/triage/rules.py
```

It combines:

- symptoms
- visual summary
- temperature
- classifier severity
- classifier confidence
- NLP risk flags
- abnormal report findings

It outputs:

- urgency
- recommended specialist
- reasons
- red flags
- next steps
- confidence policy
- rule basis
- safety note

This makes the project closer to the proposed ideology because the system is no longer just "model prediction." It now has an explicit triage decision-support layer.

## Strengths Of The Project

- Multimodal: image, webcam, symptoms, reports, and lab extraction.
- Triage-focused: safer and more defensible than diagnosis.
- Explainable: shows reasons, red flags, and confidence policy.
- Layered: classifier, VLM, NLP extractor, reasoning summary, and triage rules each have separate roles.
- Conference-ready story: helps prioritize care instead of replacing doctors.

## Current Limitations

- Not clinically validated.
- Triage rules are still prototype-level.
- Skin classifier confidence may be low on real-world images.
- Webcam/image quality affects results.
- No formal sensitivity, specificity, or clinician-labeled evaluation yet.
- Temperature handling should be improved to support both Celsius and Fahrenheit clearly.
- VLM descriptions can be imperfect and should be reviewed with the original image.

## Questions They May Ask

Prepare answers for:

- Is this a diagnostic tool?
- How accurate is your classifier?
- How was the dataset collected?
- Does it work across different skin tones?
- How do you handle false negatives?
- What happens when the model is uncertain?
- Why use rule-based NLP?
- Did doctors validate the triage rules?
- How do you protect patient data?
- Why should the system recommend a specialist?

## Best Answers

Diagnosis question:

> We do not claim diagnosis. The system supports triage by identifying visible warning signs, extracted symptoms, abnormal report values, and specialist direction.

Validation question:

> This project is currently a prototype. The contribution is the multimodal triage architecture. Clinical validation against clinician-labeled cases is future work.

Low confidence question:

> Low confidence is intentionally escalated to review rather than forced into a diagnosis. In medical triage, uncertainty should be handled conservatively.

Model role question:

> Each model has a different role: the classifier gives probability hints, the VLM describes visible findings, the NLP extractor structures symptoms and reports, and the triage layer converts those signals into urgency and specialist recommendation.

## Final Conference Position

Use this sentence:

> DiagnoBot is a multimodal medical triage decision-support prototype that combines skin/wound image observations, symptom input, and report extraction to provide explainable urgency and specialist referral recommendations. It is not a diagnostic system.

