import json
labeled_data = []
with open(r"emails_labeled.jsonl", "r") as read_file:
    for line in read_file:
        data = json.loads(line)
        labeled_data.append(data)
        
TRAINING_DATA = []
for entry in labeled_data:
    entities = []
    for e in entry['labels']:
        entities.append((e[0], e[1],e[2]))
    spacy_entry = (entry['text'], {"entities": entities})
    TRAINING_DATA.append(spacy_entry)        
        
import spacy
import random
import json
nlp = spacy.blank("en")        
        
ner = nlp.add_pipe("ner")
nlp.pipe_names        
        
ner.add_label("OIL")

# Start the training
nlp.begin_training()        
        
from spacy.training import Example
import spacy
# Loop for 40 iterations
for itn in range(40):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
    batches = spacy.util.minibatch(TRAINING_DATA, size=2)
    for batch in batches:
        examples = []
        for text, annots in batch:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        nlp.update(examples, losses=losses, drop=0.3)
    print("Losses", losses)        
        
import pickle

pickle.dump(nlp, open('nlp.pkl', 'wb'))        
        
        
        
        
        
        
        
        