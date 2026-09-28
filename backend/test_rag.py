from retrieve import answer_question

questions = [
    "Tell me about Bharat's education.",
    "What is Bharat's CGPA?",
    "Tell me about all of Bharat's projects.",
    "What internship did Bharat do?",
    "What technologies does Bharat know?",
    "Tell me about Bharat's certifications."
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    answer = answer_question(question)

    print("\nANSWER:")
    print(answer)