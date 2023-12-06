import json
from better_profanity import profanity
from profanity_filter import ProfanityFilter
from profanity_check import predict
import spacy
import csv

nlp = spacy.load('en_core_web_sm')
profanity_filter = ProfanityFilter(nlps={'en': nlp})  # reuse spacy Language (optional)
nlp.add_pipe(profanity_filter.spacy_component, last=True)

# Load the 'HS' JSON with lists
with open('HateSpeechCategoryDict.json', 'r') as hs_file:
    hs_data = json.load(hs_file)

# Open the original JSONL file for reading
with open('test.jsonl', 'r', encoding='utf-8') as file:
    results = []
    words_in_has_class = []

    # Process each line as a separate JSON entry
    for line in file:
        entry = json.loads(line)

        text = entry.get('text', '')  # Assuming 'text' is the key in your JSON data
        img = entry.get('img', '')    # Assuming 'img' is the key for the 'img' value
        doc=nlp(text)
        # Check for profanity using better_profanity
        if profanity.contains_profanity(text)==True:
            is_profane_better = 1
        else:
            is_profane_better = 0

        # Check for profanity using profanity-filter
        if doc._.is_profane==True:
            is_profane_filter = 1
        else:
            is_profane_filter = 0

        # Check for profanity using profanity-check
        is_profanity_check=predict([text])[0]

        profanity_sum=is_profane_better+is_profane_filter+is_profanity_check
        profanity_result=0
        if profanity_sum>1:
            profanity_result=2
        elif profanity_sum==1:
            profanity_result=1
        else:
            profanity_result=0

        # Check for matches in the 'HateSpeech' JSON
        has_ethnicity = int(any(word in text.lower() for word in hs_data['Ethnicity']))
        has_religion = int(any(word in text.lower() for word in hs_data['Religion']))
        has_class = int(any(word in text.lower() for word in hs_data['Class']))
        has_gender = int(any(word in text.lower() for word in hs_data['Gender']))
        has_nationality = int(any(word in text.lower() for word in hs_data['Nationality']))
        has_disability = int(any(word in text.lower() for word in hs_data['Disability']))
        has_sex_orientation = int(any(word in text.lower() for word in hs_data['Sexual Orientation']))

        words_in_ethnicity = sum(1 for word in hs_data['Ethnicity'] if word in text.lower())
        words_in_religion = sum(1 for word in hs_data['Religion'] if word in text.lower())
        words_in_class = sum(1 for word in hs_data['Class'] if word in text.lower())
        words_in_gender = sum(1 for word in hs_data['Gender'] if word in text.lower())
        words_in_nationality = sum(1 for word in hs_data['Nationality'] if word in text.lower())
        words_in_disability = sum(1 for word in hs_data['Disability'] if word in text.lower())
        words_in_sex_orientation = sum(1 for word in hs_data['Sexual Orientation'] if word in text.lower())
        
        # Create a dictionary for the result
        result_entry = {
            'path': img,
            'img': img.split('/')[-1],
            'Has_Profanity': profanity_result,
            'Has_Ethnicity': has_ethnicity,
            'Has_Religion': has_religion,
            'Has_Class': has_class,
            'Has_Gender': has_gender,
            'Has_Nationality': has_nationality,
            'Has_Disability': has_disability,
            'Has_Sexual_Orientation': has_sex_orientation,
            'Ethnicity_Count': words_in_ethnicity,
            'Religion_Count': words_in_religion,
            'Class_Count': words_in_class,
            'Gender_Count': words_in_gender,
            'Nationality_Count': words_in_nationality,
            'Disability_Count': words_in_disability,
            'Sexual_Orientation_Count': words_in_sex_orientation
        }
        # Count the number of Trues excluding 'Has_Profanity'
        true_count = sum(result_entry[key] for key in result_entry if key.startswith("Has_") and key != "Has_Profanity")

        # Add the 'True_Count' column
        result_entry['Hate Speech Category Count'] = true_count

        results.append(result_entry)


# Save to CSV
with open('test_features.csv', 'w', newline='', encoding='utf-8') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
    writer.writeheader()
    for result_entry in results:
        writer.writerow(result_entry)
