import codecs
import json
import argparse
import regex as re
import urduhack

class QuestionGenerator(object):
    """
    Generates multiple questions given an input file
    - Sentence Segmentation
    - Tokenization
    - POS Tagger
    - NER
    """
    def __init__(self, input_file=None, output_file=None):
        """
        Arguments:
            - input_file: Text file containing Urdu text
        """
        if input_file:
            self.raw_data = codecs.open(input_file, "r", "utf-8").read()
            self.output_file = output_file
            self.sentences = []
            self.questions = {}
            urduhack.download()
            self.preprocess()
    
    def preprocess(self):
        """
        Converts text into sentences
        Removes punctuation marks from each sentence
        Uses word tokenizer to add spaces between
        joined words
        """
        self.sentences = urduhack.tokenization.sentence_tokenizer(self.raw_data)
        for idx, sentence in enumerate(self.sentences):
            self.sentences[idx] = " ".join(urduhack.tokenization.word_tokenizer(urduhack.preprocessing.remove_punctuation(sentence)))
    
    def pos_tagger(self, sentence):
        """
        Generates POS tags for each sentence
        Gives list of dtuples
        """
        return urduhack.models.pos_tagger.predict_tags(sentence)

    def ner_tagger(self, sentence):
        """
        Generates NER tags for each sentence
        Gives list of tuples
        """
        return urduhack.models.ner.predict_ner(sentence)
    
    def generate_who(self, sentence, pos_tags, ner_tags):
        """
        Question type: Who
        Returns a dictionary contianing data
        """
        question_data = {}
        question = []
        question_data["type"] = "who"
        subject = ""
        object = ""
        skip = False
        for idx in range(len(pos_tags)):
            if ner_tags[idx][1] == "Person":
                if subject == "":
                    found = False
                    for i in range(idx + 1, len(pos_tags)):
                        if found and (pos_tags[i][1] == "PROPN" or pos_tags[i][1] == "NOUN"):
                            subject = "کس"
                        elif pos_tags[i][1] == "ADP":
                            if pos_tags[i][0] == "سے":
                                break
                            found = True
                    if subject == "":
                        subject = "کون"
                        skip = True
                else:
                    if skip:
                        object = pos_tags[idx][0]
                    else:
                        question.append(pos_tags[idx][0])
            elif subject != "":
                skip = False
                question.append(pos_tags[idx][0])

        question_data["question"] = subject + " " + object + " " + " ".join(question)
        question_data["answer"] = sentence
        return question_data
    
    def generate_where(self, sentence, pos_tags, ner_tags):
        """
        Question type: Where
        Returns a dictionary containing data
        """
        question_data = {}
        question = []
        question_data["type"] = "where"
        skip = False
        verb = False
        object_idx = -1
        for idx in range(len(pos_tags)):
            if ner_tags[idx][1] == "Location" and pos_tags[idx][1] == "PROPN":
                object_idx = idx

        for idx in range(len(pos_tags)):
            if object_idx == idx:
                question.append("کہاں")
                skip = True
            elif pos_tags[idx][1] == "VERB":
                question.append(pos_tags[idx][0])
                verb = True
            elif not skip:
                question.append(pos_tags[idx][0])
            elif pos_tags[idx][1] == "AUX":
                question.append(pos_tags[idx][0])
        if not verb:
            if "Person" in dict(ner_tags).values():
                question.append("گئے")
            else:
                question.append("ہے")
        
        question_data["question"] = " ".join(question)
        question_data["answer"] = sentence
        return question_data
    
    def generate_when(self, sentence, pos_tags, ner_tags):
        """
        Question type: When
        Returns a dictionary containing data
        """
        question_data = {}
        question = []
        question_data["type"] = "when"
        skip = False
        found = False
        for idx in range(len(pos_tags)):
            if ner_tags[idx][1] == "Date":
                if len(question) > 1:
                    skip = True
                question.append("کب")
                found = True
            elif pos_tags[idx][1] == "VERB":
                question.append(pos_tags[idx][0])
            elif not skip:
                if found and pos_tags[idx][1] != "ADP":
                    question.append(pos_tags[idx][0])
                if not found:
                    question.append(pos_tags[idx][0])

        if found and skip:
            question.append("ہے")

        question_data["question"] = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", " ".join(question))
        question_data["answer"] = sentence
        return question_data

    def generate_how_many(self, sentence, pos_tags, ner_tags):
        """
        Question type: How many
        Returns a dictionary containing data
        """
        question_data = {}
        question = []
        question_data["type"] = "how many"
        for idx in range(len(pos_tags)):
            if pos_tags[idx][1] == "NUM":
                question.append("کتنے")
            else:
                question.append(pos_tags[idx][0])

        question_data["question"] = " ".join(question)
        question_data["answer"] = sentence
        return question_data
    
    def generate_what(self, sentence, pos_tags, ner_tags):
        """
        Question type: What
        Returns a dictionary containing data
        """
        question_data = {}
        question = []
        question_data["type"] = "what"
        object_idx = -1
        if "Person" in dict(ner_tags).values():
            for idx in range(len(pos_tags)):
                if idx < len(pos_tags) + 1 and ner_tags[idx][1] == "Person" and pos_tags[idx + 1][0] == "نے":
                    question.append(pos_tags[idx][0])
                    question.append("نے")
                    question.append("کیا کیا")
                    question_data["question"] = " ".join(question)
                    question_data["answer"] = sentence
                    return question_data

        for idx in range(len(pos_tags)):
            if pos_tags[idx][1] == "NOUN" or pos_tags[idx][1] == "PROPN":
                object_idx = idx

        for idx in range(len(pos_tags)):
            if object_idx == idx:
                if "VERB" not in dict(pos_tags).values():
                    question.append("کیا")
            elif pos_tags[idx][1] == "ADJ":
                continue
            elif idx < len(pos_tags) - 1 and pos_tags[idx][1] == "CCONJ" and pos_tags[idx - 1][1] == "ADJ":
                continue
            elif pos_tags[idx][1] == "VERB" and idx < len(pos_tags) - 1:
                question.append("کیا کر")
            elif pos_tags[idx][1] == "VERB":
                question.append("کیا کیا")
            else:
                question.append(pos_tags[idx][0])
        
        question_data["question"] = " ".join(question)
        question_data["answer"] = sentence
        return question_data

    def generate_how(self, sentence, pos_tags, ner_tags):
        """
        Question type: How
        Returns a dictionary containing data
        """
        question_data = {}
        question = []
        question_data["type"] = "how"
        found = False
        for idx in range(len(pos_tags)):
            if not found and pos_tags[idx][1] == "ADJ":
                found = True
                if "Location" in dict(ner_tags[idx + 1:]).values():
                    question.append("کیسا")
                elif pos_tags[idx][0][-1] == "ا":
                    question.append("کیسا")
                elif pos_tags[idx][0][-1] == "ی":
                    question.append("کیسی")
                else:
                    question.append("کیا")
            elif found and (pos_tags[idx][1] == "NOUN" or pos_tags[idx][1] == "PROPN"):
                question.append(pos_tags[idx][0])
                found = False
            elif found and (pos_tags[idx][1] == "VERB" or pos_tags[idx][1] == "AUX"):
                question.append(pos_tags[idx][0])
            elif not found:
                question.append(pos_tags[idx][0])
        
        question_data["question"] = " ".join(question)
        question_data["answer"] = sentence
        return question_data
    
    def p_save(self, questions):
        """
        Saves the created dataset in an outputfile
        """
        output = codecs.open(self.output_file, "w", "utf-8")
        for data in questions:
            for key, value in data.items():
                output.write(key + ": " + value + "\n")
            output.write("---------------------------------------\n\n")
        output.close()

    def generate_questions(self):
        """
        Generates different types of questions for 
        the input text
        """
        questions = []
        for sentence in self.sentences:
            pos_tags = self.pos_tagger(sentence)
            ner_tags = self.ner_tagger(sentence)

            if "Person" in dict(ner_tags).values():
                questions.append(self.generate_who(sentence, pos_tags, ner_tags))
                if "Location" not in dict(ner_tags).values() and "Date" not in dict(ner_tags).values():
                    questions.append(self.generate_what(sentence, pos_tags, ner_tags))
            if "Location" in dict(ner_tags).values():
                question = self.generate_where(sentence, pos_tags, ner_tags)
                if question["question"] != " ".join(dict(pos_tags).keys()) + " " + "ہے":
                    ques_pos = self.pos_tagger(" ".join(question["question"]))
                    if "NOUN" in dict(ques_pos).values() or "PROPN" not in dict(ques_pos).values():
                        questions.append(question)
            if "Date" in dict(ner_tags).values():
                questions.append(self.generate_when(sentence, pos_tags, ner_tags))
            if "NUM" in dict(pos_tags).values():
                questions.append(self.generate_how_many(sentence, pos_tags, ner_tags))
            if "ADJ" in dict(pos_tags).values():
                questions.append(self.generate_how(sentence, pos_tags, ner_tags))
        
        self.p_save(questions)


arg_parser = argparse.ArgumentParser(description='Normalize all ground truth texts for the given text files.')
arg_parser.add_argument("-i", "--input_file", help="Raw text file for question generation")
arg_parser.add_argument("-o", "--output_file", help="Name of the output file")

args = arg_parser.parse_args()

generator = QuestionGenerator(args.input_file, args.output_file)
generator.generate_questions()
