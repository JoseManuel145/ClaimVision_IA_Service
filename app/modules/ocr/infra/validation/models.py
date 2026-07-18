from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    aspect_ratio: float = 0.0
    sharpness: float = 0.0
    brightness_mean: float = 0.0

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)
