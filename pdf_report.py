from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


def generate_pdf(prediction, confidence, report_lines, overall):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(Paragraph("Credit Score Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # Prediction
    content.append(Paragraph(f"Predicted Score: {prediction}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Report
    content.append(Paragraph("Detailed Analysis:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for line in report_lines:
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 8))

    # Final Advice
    content.append(Spacer(1, 12))
    content.append(Paragraph("Final Advice:", styles["Heading2"]))
    content.append(Paragraph(overall, styles["Normal"]))

    doc.build(content)

    buffer.seek(0)

    return buffer