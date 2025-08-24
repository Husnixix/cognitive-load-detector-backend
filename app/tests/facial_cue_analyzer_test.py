from app.model.facial_cue_analyzer import FacialCueAnalyzer

analyzer = FacialCueAnalyzer()
analyzer.start_facial_cue_detector()
face = analyzer.facial_cues_data
print("Print: ", face)