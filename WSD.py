from nltk.corpus import wordnet as wn, stopwords
from nltk import word_tokenize
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import senseval as se
from sklearn.model_selection import ParameterGrid


class WSD(object):

    def __init__(self, similarity_measurement, best_selector, tokenizer=word_tokenize, normalisation=None,
                 set_of_words_extractor=None, words_filter=None, stop_words=set(stopwords.words('english'))):

        self.set_of_words_extractor = set_of_words_extractor
        self.words_filter = words_filter
        self.tokenizer = tokenizer
        self.normalisation = normalisation
        self.similarity_measurement = similarity_measurement
        self.best_selector = best_selector
        self.stop_words = stop_words

    def preprocess_definition(self, definition):
        words = self.tokenizer(definition)
        words = [w for w in words if w.isalpha()]
        if self.words_filter is not None:
            words = self.words_filter(words, self.stop_words)
        if self.normalisation is not None:
            words = [self.normalisation(w) for w in words]

        return words

    def preprocess_context(self, words):
        words = [w.lower() for w in words]
        words = [w for w in words if w.isalpha()]

        if self.words_filter is not None:
            words = self.words_filter(words, self.stop_words)

        if self.normalisation is not None:
            words = [self.normalisation(w) for w in words if self.normalisation(w) is not None]

        if self.set_of_words_extractor is not None:
            words = self.set_of_words_extractor(words)

        return words

    @staticmethod
    def remove_stop_words(words, stop_words):

        return [w for w in words if w not in stop_words]

    @staticmethod
    def one_synonym_generator(words):
        answer = []
        answer += words
        for w in words:
            for ss in wn.synsets(w):
                answer += [ss.lemmas()[0].name()]
        return list(set(answer))

    @staticmethod
    def synonyms_generator(words):
        answer = []
        answer += words
        for w in words:
            for ss in wn.synsets(w):
                answer += ss.lemma_names()

        return list(set(answer))

    @staticmethod
    def jaccard_similarity(context_words, definition_words):
        A = set(context_words)
        B = set(definition_words)
        return len(A.intersection(B)) / len(A.union(B))

    @staticmethod
    def matched_words_similarity(context_words, definition_words):
        A = set(context_words)
        B = set(definition_words)
        return len(A.intersection(B))

    @staticmethod
    def select_first(definitions, context_words, similarity_measurement):
        return 0

    @staticmethod
    def select_best(definitions, context_words, similarity_measurement):
        best_matched = 0
        max_match = 0
        for i in range(len(definitions)):
            set_of_definition_words = definitions[i]
            if similarity_measurement(set_of_definition_words, context_words) > max_match:
                best_matched = i
                max_match = similarity_measurement(set_of_definition_words, context_words)
        return best_matched

    @staticmethod
    def select_best_five(definitions, context_words, similarity_measurement):
        best_matched = 0
        max_match = 0
        for i in range(min(5, len(definitions))):
            set_of_definition_words = definitions[i]
            if similarity_measurement(set_of_definition_words, context_words) > max_match:
                best_matched = i
                max_match = similarity_measurement(set_of_definition_words, context_words)
        return best_matched

    def disambiguate(self, ambiguous_word, context, pos):

        set_of_context_words = self.preprocess_context(context)
        lemmas = wn.lemmas(ambiguous_word, pos)
        definitions = []
        ambiguous_word = wn.morphy(ambiguous_word, pos)
        for i in range(len(lemmas)):
            lemma = lemmas[i]
            set_of_definition_words = self.preprocess_definition(lemma.synset().definition())
            for example in lemma.synset().examples():
                set_of_definition_words += self.preprocess_definition(example)
            set_of_definition_words = list(set(set_of_definition_words))
            definitions.append(set_of_definition_words)
        best_matched = self.best_selector(definitions, set_of_context_words, self.similarity_measurement)

        return ambiguous_word.upper() + str(best_matched + 1)


x = se.instances("hard.pos")
ex = x[20:220] + x[3500:3550] + x[4100:4150]


def evaluate(wsd):
    sum = 0

    for sent in ex:

        correct_sense = sent.senses[0]
        context = [w[0] for w in sent.context]
        word = sent.context[sent.position][0]

        sense = wsd.disambiguate(word, context, sent.word.split("-")[1])
        if sense == correct_sense:
            sum += 1

    return sum / len(ex)


param_grid = {'similarity_measurement': [WSD.jaccard_similarity, WSD.matched_words_similarity],
              'best_selector': [WSD.select_best, WSD.select_best_five],
              'tokenizer': [word_tokenize, casual_tokenize],
              'normalisation': [PorterStemmer().stem, wn.morphy, None, ],
              'set_of_words_extractor': [WSD.synonyms_generator, WSD.one_synonym_generator, None],
              'words_filter': [WSD.remove_stop_words, None]}

grid = ParameterGrid(param_grid)


def statistic(accuracy, grid):
    sum = {}
    num = {}

    for i in range(len(grid)):
        for params in grid[i]:
            value = grid[i][params]
            param = params
            if value is None:
                param += " None"
            else:
                param += " " + value.__name__

            if param not in sum:
                sum[param] = accuracy[i]
                num[param] = 1
            else:
                sum[param] += accuracy[i]
                num[param] += 1

    for param in sum:
        sum[param] /= num[param]

    print(sum)


def print_params(params):
    output_string = "\n-----------------------\n"
    for param in params:
        if params[param] is None:
            output_string += param + " == " + "NONE" + "\n"
        else:
            output_string += param + " == " + params[param].__name__ + "\n"
    output_string += "___________________"
    print(output_string)


accuracies = []
for g in grid:
    wsd = WSD(**g)
    print_params(g)
    accuracy = evaluate(wsd)
    print(accuracy)
    accuracies.append(accuracy)
print("Average accuracy of parameters")
statistic(accuracies, grid)
print(evaluate(WSD(tokenizer=casual_tokenize, best_selector=WSD.select_first,
                   similarity_measurement=WSD.matched_words_similarity)), " Accuracy by selecting common sense")
