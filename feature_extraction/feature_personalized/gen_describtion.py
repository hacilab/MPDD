import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import json
from transformers import AutoTokenizer, AutoModel

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------加载模型-----------------#
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()
# ---------------------------------#


def generate_patient_prompt(patient_data):
    """
    Generate a structured prompt for personalized description based on patient data.
    """
    # Extract relevant information
    big5_scores = patient_data.get("big5_traits", {})
    age = patient_data.get("age", "unknown")
    gender = patient_data.get("gender", "unknown")
    native_place = patient_data.get("native_place", "unknown")
    # financial_stress = patient_data.get("family_factors", {}).get("Financial_Stress", "unknown")
    # family_members = patient_data.get("family_factors", {}).get("Family_Members", "unknown")
    # disease = patient_data.get("disease", "unknown")

    # Interpret scores
    extroversion = big5_scores.get("Extraversion", "unknown")
    agreeableness = big5_scores.get("Agreeableness", "unknown")
    openness = big5_scores.get("Openness", "unknown")
    neuroticism = big5_scores.get("Neuroticism", "unknown")
    conscientiousness = big5_scores.get("Conscientiousness", "unknown")

    # Explain financial stress and disease
    # financial_stress_desc = {
    #     0: "no financial stress",
    #     1: "mild financial stress",
    #     2: "moderate financial stress",
    #     3: "severe financial stress"
    # }.get(financial_stress, "unknown financial stress level")

    # disease_desc = {
    #     "0": "the patient is healthy",
    #     "1": "the patient has other diseases",
    #     "2": "the patient has endocrine diseases",
    #     "3": "the patient has circulatory system diseases",
    #     "4": "the patient has neurological diseases"
    # }.get(disease, "unknown disease status")

    # Construct the prompt
    prompt = (
        f"The patient is a {age}-year-old {gender} from {native_place}. "
        f"The patient's Extraversion score is {extroversion}. "
        f"The Agreeableness score is {agreeableness}. "
        f"The Openness score is {openness}. "
        f"The Neuroticism score is {neuroticism}. "
        f"The Conscientiousness score is {conscientiousness}. "
        # f"Their financial stress is categorized as {financial_stress_desc}, and they live with {family_members} family members. "
        # f"Based on the disease classification, {disease_desc}. "
        "Please generate a concise, fluent English description summarizing the patient's key personality traits, family environment, and other notable characteristics. "
        "Avoid mentioning depression or related terminology. "
        "Output the response as a single paragraph."
    )

    return prompt


def process_dataset(json_file, output_file):
    """
    Process the JSON dataset and generate personalized descriptions.
    """
    with open(json_file, "r") as f:
        dataset = json.load(f)

    # Initialize the dictionary to store results
    results = {}

    # Open the output file in write mode
    with open(output_file, "w") as f:
        for patient_id, patient_data in dataset.items():
            print(f"Processing patient ID: {patient_id}")
            patient_prompt = generate_patient_prompt(patient_data)
            print(f"Generated prompt for patient {patient_id}: {patient_prompt}")

            # Use model.chat to generate personalized response
            response, history = model.chat(tokenizer, patient_prompt, history=[], temperature=0.1)
            print(f"Generated description for patient {patient_id}: {response}")

            # Store the result in the dictionary
            results[patient_id] = response

            # Write the current results to the JSON file
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")  # Add a newline for better readability in the JSON file

    print(f"All patient descriptions saved to {output_file}.")


if __name__ == "__main__":
    # Path to the dataset
    json_file = "./GLM_data/label_data.json"
    # Output description file
    output_file = "./GLM_data/personalized_descriptions.json"

    process_dataset(json_file, output_file)
