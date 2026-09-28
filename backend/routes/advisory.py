from fastapi import APIRouter

router = APIRouter(prefix="/advisory", tags=["advisory"])


GENERIC_PROTOCOL = {
    "medicine_name": "Consult local Krishi Vigyan Kendra",
    "dosage_per_acre": "As advised by local extension officer",
    "organic_alternative": "Use sanitation and crop rotation",
    "application_timing": "Follow local agronomy advice",
}


def _normalize(disease_name: str) -> str:
    """'Potato___Late_blight' -> 'late blight' (lower-cased for lookup)."""
    name = disease_name.split("___")[-1].replace("_", " ")
    return name.strip().lower() if name else disease_name.lower()


TREATMENT_PROTOCOLS = {
    # ── Tomato diseases (PlantVillage 10-class) ───────────────────────────
    "late blight": {
        "medicine_name": "Mancozeb 75% WP",
        "dosage_per_acre": "2 kg/acre in 200 L water",
        "organic_alternative": "Neem oil spray (3 mL/L)",
        "application_timing": "Apply every 7-10 days; stop 3 days before harvest",
    },
    "early blight": {
        "medicine_name": "Copper oxychloride 50% WP",
        "dosage_per_acre": "1.5 kg/acre",
        "organic_alternative": "Bacillus subtilis biofungicide",
        "application_timing": "First spray at symptom appearance, repeat every 10 days",
    },
    "leaf mold": {
        "medicine_name": "Chlorothalonil 75% WP",
        "dosage_per_acre": "1.5 kg/acre",
        "organic_alternative": "Improve greenhouse ventilation; remove infected leaves",
        "application_timing": "Spray at first sign, repeat after 10 days",
    },
    "bacterial spot": {
        "medicine_name": "Copper hydroxide 77% WP",
        "dosage_per_acre": "1 kg/acre",
        "organic_alternative": "Copper-free biofungicide; avoid overhead irrigation",
        "application_timing": "Apply at seedling stage and after every rain event",
    },
    "septoria leaf spot": {
        "medicine_name": "Chlorothalonil 75% WP",
        "dosage_per_acre": "1.2 kg/acre",
        "organic_alternative": "Neem oil; remove and destroy lower infected leaves",
        "application_timing": "Spray every 10-14 days from first sign",
    },
    "spider mites two-spotted spider mite": {
        "medicine_name": "Abamectin 1.8% EC",
        "dosage_per_acre": "200 mL/acre",
        "organic_alternative": "Neem oil + insecticidal soap spray",
        "application_timing": "Spray undersides of leaves; repeat after 7 days",
    },
    "target spot": {
        "medicine_name": "Azoxystrobin 23% SC",
        "dosage_per_acre": "200 mL/acre",
        "organic_alternative": "Trichoderma viride biocontrol",
        "application_timing": "Apply at first symptom; repeat every 14 days",
    },
    "tomato yellow leaf curl virus": {
        "medicine_name": "Imidacloprid 17.8% SL (controls whitefly vector)",
        "dosage_per_acre": "100 mL/acre",
        "organic_alternative": "Yellow sticky traps; reflective mulch to repel whitefly",
        "application_timing": "Apply at transplanting; remove infected plants immediately",
    },
    "tomato mosaic virus": {
        "medicine_name": "No direct cure; Imidacloprid for aphid vector control",
        "dosage_per_acre": "100 mL/acre",
        "organic_alternative": "Remove infected plants; disinfect tools with 10% bleach",
        "application_timing": "Preventive only; use virus-free certified seeds",
    },
    "healthy": {
        "medicine_name": "None required",
        "dosage_per_acre": "—",
        "organic_alternative": "Maintain balanced NPK fertilization and field sanitation",
        "application_timing": "Preventive monitoring only",
    },
}


def get_treatment_protocol(disease_name: str):
    return TREATMENT_PROTOCOLS.get(_normalize(disease_name), GENERIC_PROTOCOL)


@router.get("/protocol/{disease_name}")
def get_protocol(disease_name: str):
    return get_treatment_protocol(disease_name)
