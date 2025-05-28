LANGUAGES = {
    'en': {
        'model': 'en_core_web_sm-3.4.1',
        'keywords': {
            'course': ['course', 'class', 'lecture'],
            'exam': ['exam', 'test', 'midterm', 'final'],
            'teacher': ['teacher', 'professor', 'instructor']
        }
    },
    'tr': {
        'model': 'output/model-last',
        'keywords': {
            'course': ['ders', 'kurs'],
            'exam': ['sınav', 'vize', 'final', 'bütünleme'],
            'teacher': ['hoca', 'öğretmen', 'profesör']
        }
    }
}

UNIVERSITY_TERMS = {
    'buildings': ['201', '202', '203', '204', '206', 'UZEM', 'Seminer Odası']
}
