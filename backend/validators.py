from typing import List, Optional
from pydantic import BaseModel, validator, Field
from fastapi import HTTPException
import re


class UploadValidation(BaseModel):
    recent_only: bool = True
    recent_years: int = Field(default=2, ge=1, le=10, description="Years to filter (1-10)")

    @validator('recent_years')
    def validate_years(cls, v):
        if v < 1 or v > 10:
            raise ValueError('recent_years must be between 1 and 10')
        return v


class TrainValidation(BaseModel):
    dataset_id: str = Field(..., min_length=3, max_length=100)
    models: List[str] = Field(..., min_items=1, max_items=5)
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=256)
    use_gpu: bool = False
    denoising_method: Optional[str] = Field(None, pattern="^(none|gaussian|median|bilateral)$")
    denoising_strength: float = Field(default=0.1, ge=0.0, le=1.0)
    recent_only: bool = True
    recent_years: int = Field(default=2, ge=1, le=10)

    @validator('models')
    def validate_models(cls, v):
        valid_models = ['autoencoder', 'vae', 'transformer']
        for model in v:
            if model not in valid_models:
                raise ValueError(f'Invalid model: {model}. Valid models: {valid_models}')
        return v


class DetectValidation(BaseModel):
    dataset_id: str = Field(..., min_length=3, max_length=100)
    model_name: str = Field(..., pattern="^(autoencoder|vae|transformer)$")
    epochs: int = Field(default=50, ge=1, le=1000)
    threshold_percentile: float = Field(default=95.0, ge=50.0, le=99.9)
    batch_size: int = Field(default=32, ge=1, le=256)
    use_gpu: bool = False
    denoising_method: Optional[str] = Field(None, pattern="^(none|gaussian|median|bilateral)$")
    denoising_strength: float = Field(default=0.1, ge=0.0, le=1.0)
    recent_only: bool = True
    recent_years: int = Field(default=2, ge=1, le=10)


class CrossMatchValidation(BaseModel):
    object_name: str = Field(..., min_length=2, max_length=100)

    @validator('object_name')
    def validate_object_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Object name contains invalid characters')
        return v.strip()


def validate_file_extension(filename: str, allowed_extensions: List[str] = ['.csv']):
    """Validate file extension"""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = filename.lower().split('.')[-1]
    allowed_exts = [ext.lstrip('.').lower() for ext in allowed_extensions]

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{file_ext} not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )


def validate_file_size(file_size: int, max_size_mb: int = 100):
    """Validate file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size_mb}MB"
        )


def validate_csv_content(content: bytes):
    """Basic CSV content validation"""
    try:
        content_str = content.decode('utf-8')
        lines = content_str.strip().split('\n')

        if len(lines) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV file must have at least a header and one data row"
            )

        # Check header
        header = lines[0].lower()
        if 'time' not in header or 'flux' not in header:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'time' and 'flux' columns"
            )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )