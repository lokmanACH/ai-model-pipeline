# from fastapi import FastAPI, HTTPException
# from typing import List, Any, Dict
# from pydantic import BaseModel
# from bson import ObjectId
# import motor.motor_asyncio
# import os

# # ---------------- Import modules ----------------
# from models.paddleocr import run_paddle_inference
# from models.detectron2 import run_inference, load_model
# from models.utils import download_image

# # ---------------- MongoDB ----------------
# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
# DB_NAME = os.getenv("DB_NAME", "startup")

# client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# db = client[DB_NAME]

# # ---------------- Schemas ----------------

# class LayoutRegion(BaseModel):
#     class_id: int
#     type: str
#     bbox: List[int]
#     score: float
#     lines: List[str]


# class ImageResult(BaseModel):
#     url: str
#     regions: List[LayoutRegion]


# class ProcessResult(BaseModel):
#     exam_solution_texts: List[ImageResult]
#     student_answers_texts: List[ImageResult]


# # ---------------- FastAPI ----------------

# app = FastAPI(title="Exam OCR + Layout API")


# @app.on_event("startup")
# async def startup_event():
#     load_model()


# @app.get("/")
# def health():
#     return {"status": "ok", "description": "OCR + Layout API ready"}


# # ---------------- Helpers ----------------

# def crop_bbox(img, bbox: List[int]):
#     x1, y1, x2, y2 = bbox
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

#     if x2 <= x1 or y2 <= y1:
#         return None

#     return img[y1:y2, x1:x2]


# def all_images_have_regions(images: List[dict]) -> bool:
#     if not images:
#         return False

#     for img in images:
#         if not img.get("regions") or len(img["regions"]) == 0:
#             return False

#     return True


# # ---------------- Core Pipeline ----------------

# def process_single_image(img, url: str) -> ImageResult:
#     layout_regions: List[Dict[str, Any]] = run_inference(img)

#     result_regions: List[LayoutRegion] = []

#     for region in layout_regions:
#         bbox = region["bbox"]
#         region_type = region["type"]

#         crop = crop_bbox(img, bbox)

#         if crop is None or crop.size == 0:
#             result_regions.append(LayoutRegion(
#                 class_id=region["class_id"],
#                 type=region_type,
#                 bbox=bbox,
#                 score=region["score"],
#                 lines=[]
#             ))
#             continue

#         if region_type == "figure" and False:
#             lines = []
#         else:
#             lines = run_paddle_inference(crop)

#         result_regions.append(LayoutRegion(
#             class_id=region["class_id"],
#             type=region_type,
#             bbox=bbox,
#             score=region["score"],
#             lines=lines
#         ))

#     return ImageResult(url=url, regions=result_regions)


# async def process_images(images: List[dict]) -> List[ImageResult]:
#     results: List[ImageResult] = []

#     for img_info in images:
#         url = img_info.get("url", "")
#         existing_regions = img_info.get("regions")

#         # ✅ Reuse if exists
#         if existing_regions and len(existing_regions) > 0:
#             regions = [LayoutRegion(**r) for r in existing_regions]
#             results.append(ImageResult(url=url, regions=regions))
#             continue

#         # ❌ Process if missing
#         try:
#             img = download_image(url)
#             image_result = process_single_image(img, url)
#         except Exception as e:
#             image_result = ImageResult(
#                 url=url,
#                 regions=[
#                     LayoutRegion(
#                         class_id=-1,
#                         type="error",
#                         bbox=[0, 0, 0, 0],
#                         score=0.0,
#                         lines=[f"ERROR: {str(e)}"]
#                     )
#                 ]
#             )

#         results.append(image_result)

#     return results


# # ---------------- API Endpoint ----------------

# # @app.get("/process_exam/{exam_id}/{student_answer_id}", response_model=ProcessResult)
# # async def process_exam(exam_id: str, student_answer_id: str):

# #     # Validate IDs
# #     try:
# #         exam_oid = ObjectId(exam_id)
# #         sca_oid = ObjectId(student_answer_id)
# #     except Exception:
# #         raise HTTPException(status_code=400, detail="Invalid ObjectId")

# #     # Fetch from DB
# #     exam = await db["exams"].find_one({"_id": exam_oid})
# #     if not exam:
# #         raise HTTPException(status_code=404, detail="Exam not found")

# #     sca = await db["studentclassanswers"].find_one({"_id": sca_oid})
# #     if not sca:
# #         raise HTTPException(status_code=404, detail="StudentClassAnswer not found")

# #     exam_images = exam.get("solutionImages", [])
# #     student_images = sca.get("answers", [])

# #     # ===============================
# #     # ✅ HANDLE EXAM
# #     # ===============================
# #     if all_images_have_regions(exam_images):
# #         exam_results = [
# #             ImageResult(
# #                 url=img["url"],
# #                 regions=[LayoutRegion(**r) for r in img["regions"]]
# #             )
# #             for img in exam_images
# #         ]
# #     else:
# #         exam_results = await process_images(exam_images)

# #     # ===============================
# #     # ✅ HANDLE STUDENT
# #     # ===============================
# #     if all_images_have_regions(student_images):
# #         student_results = [
# #             ImageResult(
# #                 url=img["url"],
# #                 regions=[LayoutRegion(**r) for r in img["regions"]]
# #             )
# #             for img in student_images
# #         ]
# #     else:
# #         student_results = await process_images(student_images)

# #     # ===============================
# #     # 💾 SAVE ONLY MISSING
# #     # ===============================

# #     # ---- Exam ----
# #     updated_solution_images = []
# #     for idx, img_res in enumerate(exam_results):
# #         original = exam_images[idx]

# #         if original.get("regions"):
# #             updated_solution_images.append(original)
# #         else:
# #             updated_solution_images.append({
# #                 "url": img_res.url,
# #                 "public_id": original.get("public_id", ""),
# #                 "regions": [r.dict() for r in img_res.regions]
# #             })

# #     await db["exams"].update_one(
# #         {"_id": exam_oid},
# #         {"$set": {"solutionImages": updated_solution_images}}
# #     )

# #     # ---- Student ----
# #     updated_answers = []
# #     for idx, img_res in enumerate(student_results):
# #         original = student_images[idx]

# #         if original.get("regions"):
# #             updated_answers.append(original)
# #         else:
# #             updated_answers.append({
# #                 "url": img_res.url,
# #                 "public_id": original.get("public_id", ""),
# #                 "regions": [r.dict() for r in img_res.regions]
# #             })

# #     await db["studentclassanswers"].update_one(
# #         {"_id": sca_oid},
# #         {"$set": {"answers": updated_answers}}
# #     )

# #     # ===============================
# #     # RETURN
# #     # ===============================
# #     return ProcessResult(
# #         exam_solution_texts=exam_results,
# #         student_answers_texts=student_results,
# #     )




# @app.get("/process_exam/{student_answer_id}", response_model=ProcessResult)
# async def process_exam(student_answer_id: str):

#     # Validate ID
#     try:
#         sca_oid = ObjectId(student_answer_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid ObjectId")

#     # ===============================
#     # 🔍 GET STUDENT ANSWER FIRST
#     # ===============================
#     sca = await db["studentclassanswers"].find_one({"_id": sca_oid})
#     if not sca:
#         raise HTTPException(status_code=404, detail="StudentClassAnswer not found")

#     # ✅ Extract exam_id from reference
#     exam_id = sca.get("exam")

#     if not exam_id:
#         raise HTTPException(status_code=400, detail="Exam reference missing")

#     # ===============================
#     # 🔍 GET EXAM USING REFERENCE
#     # ===============================
#     exam = await db["exams"].find_one({"_id": exam_id})
#     if not exam:
#         raise HTTPException(status_code=404, detail="Exam not found")

#     exam_images = exam.get("solutionImages", [])
#     student_images = sca.get("answers", [])

#     # ===============================
#     # ✅ HANDLE EXAM
#     # ===============================
#     if all_images_have_regions(exam_images):
#         exam_results = [
#             ImageResult(
#                 url=img["url"],
#                 regions=[LayoutRegion(**r) for r in img["regions"]]
#             )
#             for img in exam_images
#         ]
#     else:
#         exam_results = await process_images(exam_images)

#     # ===============================
#     # ✅ HANDLE STUDENT
#     # ===============================
#     if all_images_have_regions(student_images):
#         student_results = [
#             ImageResult(
#                 url=img["url"],
#                 regions=[LayoutRegion(**r) for r in img["regions"]]
#             )
#             for img in student_images
#         ]
#     else:
#         student_results = await process_images(student_images)

#     # ===============================
#     # 💾 SAVE ONLY MISSING
#     # ===============================

#     # ---- Save Exam ----
#     updated_solution_images = []
#     for idx, img_res in enumerate(exam_results):
#         original = exam_images[idx]

#         if original.get("regions"):
#             updated_solution_images.append(original)
#         else:
#             updated_solution_images.append({
#                 "url": img_res.url,
#                 "public_id": original.get("public_id", ""),
#                 "regions": [r.dict() for r in img_res.regions]
#             })

#     await db["exams"].update_one(
#         {"_id": exam_id},
#         {"$set": {"solutionImages": updated_solution_images}}
#     )

#     # ---- Save Student ----
#     updated_answers = []
#     for idx, img_res in enumerate(student_results):
#         original = student_images[idx]

#         if original.get("regions"):
#             updated_answers.append(original)
#         else:
#             updated_answers.append({
#                 "url": img_res.url,
#                 "public_id": original.get("public_id", ""),
#                 "regions": [r.dict() for r in img_res.regions]
#             })

#     await db["studentclassanswers"].update_one(
#         {"_id": sca_oid},
#         {"$set": {"answers": updated_answers}}
#     )

#     # ===============================
#     # RETURN
#     # ===============================
#     return ProcessResult(
#         exam_solution_texts=exam_results,
#         student_answers_texts=student_results,
#     )








from fastapi import FastAPI, HTTPException, Header
from typing import List, Any, Dict, Optional
from pydantic import BaseModel
from bson import ObjectId
import motor.motor_asyncio
import os

# ---------------- Import modules ----------------
from models.paddleocr import run_paddle_inference
from models.detectron2 import run_inference, load_model
from models.utils import download_image

# ---------------- MongoDB ----------------
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "startup")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# ---------------- Schemas ----------------
class LayoutRegion(BaseModel):
    class_id: int
    type: str
    bbox: List[int]
    score: float
    lines: List[str]


class ImageResult(BaseModel):
    url: str
    regions: List[LayoutRegion]


class ProcessResult(BaseModel):
    exam_solution_texts: List[ImageResult]
    student_answers_texts: List[ImageResult]

# ---------------- FastAPI ----------------
app = FastAPI(title="Exam OCR + Layout API")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def health():
    return {"status": "ok", "description": "OCR + Layout API ready"}

# ---------------- Helpers ----------------
def crop_bbox(img, bbox: List[int]):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def all_images_have_regions(images: List[dict]) -> bool:
    if not images:
        return False
    for img in images:
        if not img.get("regions") or len(img["regions"]) == 0:
            return False
    return True

# ---------------- Core Pipeline ----------------
def process_single_image(img, url: str) -> ImageResult:
    layout_regions: List[Dict[str, Any]] = run_inference(img)
    result_regions: List[LayoutRegion] = []

    for region in layout_regions:
        bbox = region["bbox"]
        region_type = region["type"]
        crop = crop_bbox(img, bbox)

        if crop is None or crop.size == 0:
            result_regions.append(LayoutRegion(
                class_id=region["class_id"],
                type=region_type,
                bbox=bbox,
                score=region["score"],
                lines=[]
            ))
            continue

        lines = run_paddle_inference(crop) if region_type != "figure" else []
        result_regions.append(LayoutRegion(
            class_id=region["class_id"],
            type=region_type,
            bbox=bbox,
            score=region["score"],
            lines=lines
        ))

    return ImageResult(url=url, regions=result_regions)

async def process_images(images: List[dict]) -> List[ImageResult]:
    results: List[ImageResult] = []
    for img_info in images:
        url = img_info.get("url", "")
        existing_regions = img_info.get("regions")

        if existing_regions and len(existing_regions) > 0:
            regions = [LayoutRegion(**r) for r in existing_regions]
            results.append(ImageResult(url=url, regions=regions))
            continue

        try:
            img = download_image(url)
            image_result = process_single_image(img, url)
        except Exception as e:
            image_result = ImageResult(
                url=url,
                regions=[LayoutRegion(
                    class_id=-1,
                    type="error",
                    bbox=[0, 0, 0, 0],
                    score=0.0,
                    lines=[f"ERROR: {str(e)}"]
                )]
            )

        results.append(image_result)
    return results

# ---------------- API Endpoint ----------------
@app.get("/process_exam/{student_answer_id}", response_model=ProcessResult)
async def process_exam(
    student_answer_id: str,
    x_api_key: Optional[str] = Header(None)  # receive token in request header
):

    if not x_api_key:
        raise HTTPException(status_code=400, detail="API token missing in header 'x-api-key'")

    # 🔒 Now you can create your Gemini/OpenAI client here using this token
    # Example:
    # from google import genai
    # client = genai.Client(api_key=x_api_key)

    # ------------------- Validate ID -------------------
    try:
        sca_oid = ObjectId(student_answer_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")

    sca = await db["studentclassanswers"].find_one({"_id": sca_oid})
    if not sca:
        raise HTTPException(status_code=404, detail="StudentClassAnswer not found")

    exam_id = sca.get("exam")
    if not exam_id:
        raise HTTPException(status_code=400, detail="Exam reference missing")

    exam = await db["exams"].find_one({"_id": exam_id})
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    exam_images = exam.get("solutionImages", [])
    student_images = sca.get("answers", [])

    # ------------------- Process -------------------
    exam_results = exam_images if all_images_have_regions(exam_images) else await process_images(exam_images)
    student_results = student_images if all_images_have_regions(student_images) else await process_images(student_images)

    # ------------------- Save missing regions -------------------
    updated_solution_images = []
    for idx, img_res in enumerate(exam_results):
        original = exam_images[idx]
        if original.get("regions"):
            updated_solution_images.append(original)
        else:
            updated_solution_images.append({
                "url": img_res.url,
                "public_id": original.get("public_id", ""),
                "regions": [r.dict() for r in img_res.regions]
            })

    await db["exams"].update_one(
        {"_id": exam_id},
        {"$set": {"solutionImages": updated_solution_images}}
    )

    updated_answers = []
    for idx, img_res in enumerate(student_results):
        original = student_images[idx]
        if original.get("regions"):
            updated_answers.append(original)
        else:
            updated_answers.append({
                "url": img_res.url,
                "public_id": original.get("public_id", ""),
                "regions": [r.dict() for r in img_res.regions]
            })

    await db["studentclassanswers"].update_one(
        {"_id": sca_oid},
        {"$set": {"answers": updated_answers}}
    )

    return ProcessResult(
        exam_solution_texts=exam_results,
        student_answers_texts=student_results,
    )







