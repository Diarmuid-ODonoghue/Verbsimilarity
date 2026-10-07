import numpy as np
from sense2vec import Sense2Vec # word2vec from the "Gensim" package.
from nltk.corpus import wordnet
from random import random
import sys
import os
import re
from sense2vec import Sense2Vec
from nltk.corpus import wordnet as wn  # names, stopwords, words
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet_ic
# import word2vec

wnl = WordNetLemmatizer()

wn_lemmas = set(wordnet.all_lemma_names())

global LCSLlist
LCSLlist = []

#homePath = os.environ['HOME']
#s2v= Sense2Vec().from_disk(homePath+'/.local/lib/python3.8/site-packages/sense2vec/tests/data/')
s2v = Sense2Vec().from_disk("C:/Users/dodonoghue/Documents/Python-Me/Sense2Vec/s2v_reddit_2019_lg/")
print("Sense2vec is finally loaded. ")  # doc[1]._.s2v_most_similar(10))

# query = "apple|NOUN"
# vector = s2v[query]

# 3,000 most common words
k3_words = ['a', 'abandon', 'ability', 'able', 'abortion', 'about', 'above', 'abroad', 'absence', 'absolute',
            'absolutely', 'absorb', 'abuse', 'academic', 'accept', 'access',
            'accident', 'accompany', 'accomplish', 'according', 'account', 'accurate', 'accuse', 'achieve',
            'achievement', 'acid', 'acknowledge', 'acquire', 'across', 'act',
            'action', 'active', 'activist', 'activity', 'actor', 'actress', 'actual', 'actually', 'ad', 'adapt', 'add',
            'addition', 'additional', 'address', 'adequate', 'adjust',
            'adjustment', 'administration', 'administrator', 'admire', 'admission', 'admit', 'adolescent', 'adopt',
            'adult', 'advance', 'advanced', 'advantage', 'adventure',
            'advertising', 'advice', 'advise', 'adviser', 'advocate', 'affair', 'affect', 'afford', 'afraid', 'African',
            'African-American', 'after', 'afternoon', 'again', 'against',
            'age', 'agency', 'agenda', 'agent', 'aggressive', 'ago', 'agree', 'agreement', 'agricultural', 'ah',
            'ahead', 'aid', 'aide', 'AIDS', 'aim', 'air', 'aircraft', 'airline',
            'airport', 'album', 'alcohol', 'alive', 'all', 'alliance', 'allow', 'ally', 'almost', 'alone', 'along',
            'already', 'also', 'alter', 'alternative', 'although', 'always', 'AM',
            'amazing', 'American', 'among', 'amount', 'analysis', 'analyst', 'analyze', 'ancient', 'and', 'anger',
            'angle', 'angry', 'animal', 'anniversary', 'announce', 'annual',
            'another', 'answer', 'anticipate', 'anxiety', 'any', 'anybody', 'anymore', 'anyone', 'anything', 'anyway',
            'anywhere', 'apart', 'apartment', 'apparent', 'apparently', 'appeal', 'appear', 'appearance', 'apple',
            'application', 'apply', 'appoint', 'appointment', 'appreciate', 'approach', 'appropriate', 'approval',
            'approve', 'approximately', 'Arab', 'architect', 'area', 'argue', 'argument', 'arise', 'arm', 'armed',
            'army', 'around', 'arrange', 'arrangement', 'arrest', 'arrival',
            'arrive', 'art', 'article', 'artist', 'artistic', 'as', 'Asian', 'aside', 'ask', 'asleep', 'aspect',
            'assault', 'assert', 'assess', 'assessment', 'asset', 'assign', 'assignment', 'assist', 'assistance',
            'assistant', 'associate', 'association', 'assume', 'assumption', 'assure', 'at',
            'athlete', 'athletic', 'atmosphere', 'attach', 'attack', 'attempt', 'attend', 'attention', 'attitude',
            'attorney', 'attract', 'attractive', 'attribute', 'audience', 'author',
            'authority', 'auto', 'available', 'average', 'avoid', 'award', 'aware', 'awareness', 'away', 'awful',
            'baby', 'back', 'background', 'bad', 'badly', 'bag', 'bake', 'balance',
            'ball', 'ban', 'band', 'bank', 'bar', 'barely', 'barrel', 'barrier', 'base', 'baseball', 'basic',
            'basically', 'basis', 'basket', 'basketball', 'bathroom', 'battery', 'battle',
            'be', 'beach', 'bean', 'bear', 'beat', 'beautiful', 'beauty', 'because', 'become', 'bed', 'bedroom', 'beer',
            'before', 'begin', 'beginning', 'behavior', 'behind', 'being',
            'belief', 'believe', 'bell', 'belong', 'below', 'belt', 'bench', 'bend', 'beneath', 'benefit', 'beside',
            'besides', 'best', 'bet', 'better', 'between', 'beyond', 'Bible', 'big',
            'bike', 'bill', 'billion', 'bind', 'biological', 'bird', 'birth', 'birthday', 'bit', 'bite', 'black',
            'blade', 'blame', 'blanket', 'blind', 'block', 'blood', 'blow', 'blue',
            'board', 'boat', 'body', 'bomb', 'bombing', 'bond', 'bone', 'book', 'boom', 'boot', 'border', 'born',
            'borrow', 'boss', 'both', 'bother', 'bottle', 'bottom', 'boundary', 'bowl',
            'box', 'boy', 'boyfriend', 'brain', 'branch', 'brand', 'bread', 'break', 'breakfast', 'breast', 'breath',
            'breathe', 'brick', 'bridge', 'brief', 'briefly', 'bright', 'brilliant',
            'bring', 'British', 'broad', 'broken', 'brother', 'brown', 'brush', 'buck', 'budget', 'build', 'building',
            'bullet', 'bunch', 'burden', 'burn', 'bury', 'bus', 'business', 'busy',
            'but', 'butter', 'button', 'buy', 'buyer', 'by', 'cabin', 'cabinet', 'cable', 'cake', 'calculate', 'call',
            'camera', 'camp', 'campaign', 'campus', 'can', 'Canadian', 'cancer',
            'candidate', 'cap', 'capability', 'capable', 'capacity', 'capital', 'captain', 'capture', 'car', 'carbon',
            'card', 'care', 'career', 'careful', 'carefully', 'carrier', 'carry',
            'case', 'cash', 'cast', 'cat', 'catch', 'category', 'Catholic', 'cause', 'ceiling', 'celebrate',
            'celebration', 'celebrity', 'cell', 'center', 'central', 'century', 'CEO',
            'ceremony', 'certain', 'certainly', 'chain', 'chair', 'chairman', 'challenge', 'chamber', 'champion',
            'championship', 'chance', 'change', 'changing', 'channel', 'chapter',
            'character', 'characteristic', 'characterize', 'charge', 'charity', 'chart', 'chase', 'cheap', 'check',
            'cheek', 'cheese', 'chef', 'chemical', 'chest', 'chicken', 'chief',
            'child', 'childhood', 'Chinese', 'chip', 'chocolate', 'choice', 'cholesterol', 'choose', 'Christian',
            'Christmas', 'church', 'cigarette', 'circle', 'circumstance', 'cite',
            'citizen', 'city', 'civil', 'civilian', 'claim', 'class', 'classic', 'classroom', 'clean', 'clear',
            'clearly', 'client', 'climate', 'climb', 'clinic', 'clinical', 'clock',
            'close', 'closely', 'closer', 'clothes', 'clothing', 'cloud', 'club', 'clue', 'cluster', 'coach', 'coal',
            'coalition', 'coast', 'coat', 'code', 'coffee', 'cognitive', 'cold',
            'collapse', 'colleague', 'collect', 'collection', 'collective', 'college', 'colonial', 'color', 'column',
            'combination', 'combine', 'come', 'comedy', 'comfort', 'comfortable',
            'command', 'commander', 'comment', 'commercial', 'commission', 'commit', 'commitment', 'committee',
            'common', 'communicate', 'communication', 'community', 'company', 'compare',
            'comparison', 'compete', 'competition', 'competitive', 'competitor', 'complain', 'complaint', 'complete',
            'completely', 'complex', 'complicated', 'component', 'compose', 'composition',
            'comprehensive', 'computer', 'concentrate', 'concentration', 'concept', 'concern', 'concerned', 'concert',
            'conclude', 'conclusion', 'concrete', 'condition', 'conduct',
            'conference', 'confidence', 'confident', 'confirm', 'conflict', 'confront', 'confusion', 'Congress',
            'congressional', 'connect', 'connection', 'consciousness', 'consensus',
            'consequence', 'conservative', 'consider', 'considerable', 'consideration', 'consist', 'consistent',
            'constant', 'constantly', 'constitute', 'constitutional', 'construct',
            'construction', 'consultant', 'consume', 'consumer', 'consumption', 'contact', 'contain', 'container',
            'contemporary', 'content', 'contest', 'context', 'continue', 'continued',
            'contract', 'contrast', 'contribute', 'contribution', 'control', 'controversial', 'controversy',
            'convention', 'conventional', 'conversation', 'convert', 'conviction', 'convince',
            'cook', 'cookie', 'cooking', 'cool', 'cooperation', 'cop', 'cope', 'copy', 'core', 'corn', 'corner',
            'corporate', 'corporation', 'correct', 'correspondent', 'cost', 'cotton',
            'couch', 'could', 'council', 'counselor', 'count', 'counter', 'country', 'county', 'couple', 'courage',
            'course', 'court', 'cousin', 'cover', 'coverage', 'cow', 'crack', 'craft',
            'crash', 'crazy', 'cream', 'create', 'creation', 'creative', 'creature', 'credit', 'crew', 'crime',
            'criminal', 'crisis', 'criteria', 'critic', 'critical', 'criticism', 'criticize',
            'crop', 'cross', 'crowd', 'crucial', 'cry', 'cultural', 'culture', 'cup', 'curious', 'current', 'currently',
            'curriculum', 'custom', 'customer', 'cut', 'cycle', 'dad', 'daily',
            'damage', 'dance', 'danger', 'dangerous', 'dare', 'dark', 'darkness', 'data', 'date', 'daughter', 'day',
            'dead', 'deal', 'dealer', 'dear', 'death', 'debate', 'debt', 'decade',
            'decide', 'decision', 'deck', 'declare', 'decline', 'decrease', 'deep', 'deeply', 'deer', 'defeat',
            'defend', 'defendant', 'defense', 'defensive', 'deficit', 'define', 'definitely',
            'definition', 'degree', 'delay', 'deliver', 'delivery', 'demand', 'democracy', 'Democrat', 'democratic',
            'demonstrate', 'demonstration', 'deny', 'department', 'depend', 'dependent',
            'depending', 'depict', 'depression', 'depth', 'deputy', 'derive', 'describe', 'description', 'desert',
            'deserve', 'design', 'designer', 'desire', 'desk', 'desperate', 'despite',
            'destroy', 'destruction', 'detail', 'detailed', 'detect', 'determine', 'develop', 'developing',
            'development', 'device', 'devote', 'dialogue', 'die', 'diet', 'differ', 'difference',
            'different', 'differently', 'difficult', 'difficulty', 'dig', 'digital', 'dimension', 'dining', 'dinner',
            'direct', 'direction', 'directly', 'director', 'dirt', 'dirty', 'disability',
            'disagree', 'disappear', 'disaster', 'discipline', 'discourse', 'discover', 'discovery', 'discrimination',
            'discuss', 'discussion', 'disease', 'dish', 'dismiss', 'disorder', 'display',
            'dispute', 'distance', 'distant', 'distinct', 'distinction', 'distinguish', 'distribute', 'distribution',
            'district', 'diverse', 'diversity', 'divide', 'division', 'divorce', 'DNA',
            'do', 'doctor', 'document', 'dog', 'domestic', 'dominant', 'dominate', 'door', 'double', 'doubt', 'down',
            'downtown', 'dozen', 'draft', 'drag', 'drama', 'dramatic', 'dramatically',
            'draw', 'drawing', 'dream', 'dress', 'drink', 'drive', 'driver', 'drop', 'drug', 'dry', 'due', 'during',
            'dust', 'duty', 'each', 'eager', 'ear', 'early', 'earn', 'earnings', 'earth',
            'ease', 'easily', 'east', 'eastern', 'easy', 'eat', 'economic', 'economics', 'economist', 'economy', 'edge',
            'edition', 'editor', 'educate', 'education', 'educational', 'educator',
            'effect', 'effective', 'effectively', 'efficiency', 'efficient', 'effort', 'egg', 'eight', 'either',
            'elderly', 'elect', 'election', 'electric', 'electricity', 'electronic', 'element',
            'elementary', 'eliminate', 'elite', 'else', 'elsewhere', 'e-mail', 'embrace', 'emerge', 'emergency',
            'emission', 'emotion', 'emotional', 'emphasis', 'emphasize', 'employ', 'employee',
            'employer', 'employment', 'empty', 'enable', 'encounter', 'encourage', 'end', 'enemy', 'energy',
            'enforcement', 'engage', 'engine', 'engineer', 'engineering', 'English', 'enhance', 'enjoy',
            'enormous', 'enough', 'ensure', 'enter', 'enterprise', 'entertainment', 'entire', 'entirely', 'entrance',
            'entry', 'environment', 'environmental', 'episode', 'equal', 'equally', 'equipment',
            'era', 'error', 'escape', 'especially', 'essay', 'essential', 'essentially', 'establish', 'establishment',
            'estate', 'estimate', 'etc', 'ethics', 'ethnic', 'European', 'evaluate', 'evaluation',
            'even', 'evening', 'event', 'eventually', 'ever', 'every', 'everybody', 'everyday', 'everyone',
            'everything', 'everywhere', 'evidence', 'evolution', 'evolve', 'exact', 'exactly', 'examination',
            'examine', 'example', 'exceed', 'excellent', 'except', 'exception', 'exchange', 'exciting', 'executive',
            'exercise', 'exhibit', 'exhibition', 'exist', 'existence', 'existing', 'expand',
            'expansion', 'expect', 'expectation', 'expense', 'expensive', 'experience', 'experiment', 'expert',
            'explain', 'explanation', 'explode', 'explore', 'explosion', 'expose', 'exposure',
            'express', 'expression', 'extend', 'extension', 'extensive', 'extent', 'external', 'extra', 'extraordinary',
            'extreme', 'extremely', 'eye', 'fabric', 'face', 'facility', 'fact', 'factor',
            'factory', 'faculty', 'fade', 'fail', 'failure', 'fair', 'fairly', 'faith', 'fall', 'false', 'familiar',
            'family', 'famous', 'fan', 'fantasy', 'far', 'farm', 'farmer', 'fashion', 'fast',
            'fat', 'fate', 'father', 'fault', 'favor', 'favorite', 'fear', 'feature', 'federal', 'fee', 'feed', 'feel',
            'feeling', 'fellow', 'female', 'fence', 'few', 'fewer', 'fiber', 'fiction', 'field',
            'fifteen', 'fifth', 'fifty', 'fight', 'fighter', 'fighting', 'figure', 'file', 'fill', 'film', 'final',
            'finally', 'finance', 'financial', 'find', 'finding', 'fine', 'finger', 'finish', 'fire',
            'firm', 'first', 'fish', 'fishing', 'fit', 'fitness', 'five', 'fix', 'flag', 'flame', 'flat', 'flavor',
            'flee', 'flesh', 'flight', 'float', 'floor', 'flow', 'flower', 'fly', 'focus', 'folk',
            'follow', 'following', 'food', 'foot', 'football', 'for', 'force', 'foreign', 'forest', 'forever', 'forget',
            'form', 'formal', 'formation', 'former', 'formula', 'forth', 'fortune', 'forward',
            'found', 'foundation', 'founder', 'four', 'fourth', 'frame', 'framework', 'free', 'freedom', 'freeze',
            'French', 'frequency', 'frequent', 'frequently', 'fresh', 'friend', 'friendly',
            'friendship', 'from', 'front', 'fruit', 'frustration', 'fuel', 'full', 'fully', 'fun', 'function', 'fund',
            'fundamental', 'funding', 'funeral', 'funny', 'furniture', 'furthermore', 'future',
            'gain', 'galaxy', 'gallery', 'game', 'gang', 'gap', 'garage', 'garden', 'garlic', 'gas', 'gate', 'gather',
            'gay', 'gaze', 'gear', 'gender', 'gene', 'general', 'generally', 'generate',
            'generation', 'genetic', 'gentleman', 'gently', 'German', 'gesture', 'get', 'ghost', 'giant', 'gift',
            'gifted', 'girl', 'girlfriend', 'give', 'given', 'glad', 'glance', 'glass', 'global',
            'glove', 'go', 'goal', 'God', 'gold', 'golden', 'golf', 'good', 'government', 'governor', 'grab', 'grade',
            'gradually', 'graduate', 'grain', 'grand', 'grandfather', 'grandmother', 'grant',
            'grass', 'grave', 'gray', 'great', 'greatest', 'green', 'grocery', 'ground', 'group', 'grow', 'growing',
            'growth', 'guarantee', 'guard', 'guess', 'guest', 'guide', 'guideline', 'guilty',
            'gun', 'guy', 'habit', 'habitat', 'hair', 'half', 'hall', 'hand', 'handful', 'handle', 'hang', 'happen',
            'happy', 'hard', 'hardly', 'hat', 'hate', 'have', 'he', 'head', 'headline', 'headquarters',
            'health', 'healthy', 'hear', 'hearing', 'heart', 'heat', 'heaven', 'heavily', 'heavy', 'heel', 'height',
            'helicopter', 'hell', 'hello', 'help', 'helpful', 'her', 'here', 'heritage', 'hero',
            'herself', 'hey', 'hi', 'hide', 'high', 'highlight', 'highly', 'highway', 'hill', 'him', 'himself', 'hip',
            'hire', 'his', 'historian', 'historic', 'historical', 'history', 'hit', 'hold', 'hole',
            'holiday', 'holy', 'home', 'homeless', 'honest', 'honey', 'honor', 'hope', 'horizon', 'horror', 'horse',
            'hospital', 'host', 'hot', 'hotel', 'hour', 'house', 'household', 'housing', 'how',
            'however', 'huge', 'human', 'humor', 'hundred', 'hungry', 'hunter', 'hunting', 'hurt', 'husband',
            'hypothesis', 'I', 'ice', 'idea', 'ideal', 'identification', 'identify', 'identity', 'ie',
            'if', 'ignore', 'ill', 'illegal', 'illness', 'illustrate', 'image', 'imagination', 'imagine', 'immediate',
            'immediately', 'immigrant', 'immigration', 'impact', 'implement', 'implication',
            'imply', 'importance', 'important', 'impose', 'impossible', 'impress', 'impression', 'impressive',
            'improve', 'improvement', 'in', 'incentive', 'incident', 'include', 'including', 'income',
            'incorporate', 'increase', 'increased', 'increasing', 'increasingly', 'incredible', 'indeed',
            'independence', 'independent', 'index', 'Indian', 'indicate', 'indication', 'individual',
            'industrial', 'industry', 'infant', 'infection', 'inflation', 'influence', 'inform', 'information',
            'ingredient', 'initial', 'initially', 'initiative', 'injury', 'inner', 'innocent',
            'inquiry', 'inside', 'insight', 'insist', 'inspire', 'install', 'instance', 'instead', 'institution',
            'institutional', 'instruction', 'instructor', 'instrument', 'insurance', 'intellectual',
            'intelligence', 'intend', 'intense', 'intensity', 'intention', 'interaction', 'interest', 'interested',
            'interesting', 'internal', 'international', 'Internet', 'interpret', 'interpretation',
            'intervention', 'interview', 'into', 'introduce', 'introduction', 'invasion', 'invest', 'investigate',
            'investigation', 'investigator', 'investment', 'investor', 'invite', 'involve', 'involved',
            'involvement', 'Iraqi', 'Irish', 'iron', 'Islamic', 'island', 'Israeli', 'issue', 'it', 'Italian', 'item',
            'its', 'itself', 'jacket', 'jail', 'Japanese', 'jet', 'Jew', 'Jewish', 'job', 'join', 'joint',
            'joke', 'journal', 'journalist', 'journey', 'joy', 'judge', 'judgment', 'juice', 'jump', 'junior', 'jury',
            'just', 'justice', 'justify', 'keep', 'key', 'kick', 'kid', 'kill', 'killer', 'killing',
            'kind', 'king', 'kiss', 'kitchen', 'knee', 'knife', 'knock', 'know', 'knowledge', 'lab', 'label', 'labor',
            'laboratory', 'lack', 'lady', 'lake', 'land', 'landscape', 'language', 'lap', 'large', 'largely',
            'last', 'late', 'later', 'Latin', 'latter', 'laugh', 'launch', 'law', 'lawn', 'lawsuit', 'lawyer', 'lay',
            'layer', 'lead', 'leader', 'leadership', 'leading', 'leaf', 'league', 'lean', 'learn', 'learning',
            'least', 'leather', 'leave', 'left', 'leg', 'legacy', 'legal', 'legend', 'legislation', 'legitimate',
            'lemon', 'length', 'less', 'lesson', 'let', 'letter', 'level', 'liberal', 'library', 'license', 'lie',
            'life', 'lifestyle', 'lifetime', 'lift', 'light', 'like', 'likely', 'limit', 'limitation', 'limited',
            'line', 'link', 'lip', 'list', 'listen', 'literally', 'literary', 'literature', 'little', 'live',
            'living', 'load', 'loan', 'local', 'locate', 'location', 'lock', 'long', 'long-term', 'look', 'loose',
            'lose', 'loss', 'lost', 'lot', 'lots', 'loud', 'love', 'lovely', 'lover', 'low', 'lower', 'luck',
            'lucky', 'lunch', 'lung', 'machine', 'mad', 'magazine', 'mail', 'main', 'mainly', 'maintain', 'maintenance',
            'major', 'majority', 'make', 'maker', 'makeup', 'male', 'mall', 'man', 'manage', 'management',
            'manager', 'manner', 'manufacturer', 'manufacturing', 'many', 'map', 'margin', 'mark', 'market',
            'marketing', 'marriage', 'married', 'marry', 'mask', 'mass', 'massive', 'master', 'match', 'material',
            'math',
            'matter', 'may', 'maybe', 'mayor', 'me', 'meal', 'mean', 'meaning', 'meanwhile', 'measure', 'measurement',
            'meat', 'mechanism', 'media', 'medical', 'medication', 'medicine', 'medium', 'meet', 'meeting',
            'member', 'membership', 'memory', 'mental', 'mention', 'menu', 'mere', 'merely', 'mess', 'message', 'metal',
            'meter', 'method', 'Mexican', 'middle', 'might', 'military', 'milk', 'million', 'mind', 'mine',
            'minister', 'minor', 'minority', 'minute', 'miracle', 'mirror', 'miss', 'missile', 'mission', 'mistake',
            'mix', 'mixture', 'mm-hmm', 'mode', 'model', 'moderate', 'modern', 'modest', 'mom', 'moment', 'money',
            'monitor', 'month', 'mood', 'moon', 'moral', 'more', 'moreover', 'morning', 'mortgage', 'most', 'mostly',
            'mother', 'motion', 'motivation', 'motor', 'mount', 'mountain', 'mouse', 'mouth', 'move', 'movement',
            'movie', 'Mr', 'Mrs', 'Ms', 'much', 'multiple', 'murder', 'muscle', 'museum', 'music', 'musical',
            'musician', 'Muslim', 'must', 'mutual', 'my', 'myself', 'mystery', 'myth', 'naked', 'name', 'narrative',
            'narrow', 'nation', 'national', 'native', 'natural', 'naturally', 'nature', 'near', 'nearby', 'nearly',
            'necessarily', 'necessary', 'neck', 'need', 'negative', 'negotiate', 'negotiation', 'neighbor',
            'neighborhood', 'neither', 'nerve', 'nervous', 'net', 'network', 'never', 'nevertheless', 'new', 'newly',
            'news', 'newspaper', 'next', 'nice', 'night', 'nine', 'no', 'nobody', 'nod', 'noise', 'nomination',
            'none', 'nonetheless', 'nor', 'normal', 'normally', 'north', 'northern', 'nose', 'not', 'note', 'nothing',
            'notice', 'notion', 'novel', 'now', 'nowhere', 'not', 'nuclear', 'number', 'numerous', 'nurse', 'nut',
            'object', 'objective', 'obligation', 'observation', 'observe', 'observer', 'obtain', 'obvious', 'obviously',
            'occasion', 'occasionally', 'occupation', 'occupy', 'occur', 'ocean', 'odd', 'odds', 'of', 'off',
            'offense', 'offensive', 'offer', 'office', 'officer', 'official', 'often', 'oh', 'oil', 'ok', 'okay', 'old',
            'Olympic', 'on', 'once', 'one', 'ongoing', 'onion', 'online', 'only', 'onto', 'open', 'opening',
            'operate', 'operating', 'operation', 'operator', 'opinion', 'opponent', 'opportunity', 'oppose', 'opposite',
            'opposition', 'option', 'or', 'orange', 'order', 'ordinary', 'organic', 'organization', 'organize',
            'orientation', 'origin', 'original', 'originally', 'other', 'others', 'otherwise', 'ought', 'our',
            'ourselves', 'out', 'outcome', 'outside', 'oven', 'over', 'overall', 'overcome', 'overlook', 'owe', 'own',
            'owner', 'pace', 'pack', 'package', 'page', 'pain', 'painful', 'paint', 'painter', 'painting', 'pair',
            'pale', 'Palestinian', 'palm', 'pan', 'panel', 'pant', 'paper', 'parent', 'park', 'parking', 'part',
            'participant', 'participate', 'participation', 'particular', 'particularly', 'partly', 'partner',
            'partnership', 'party', 'pass', 'passage', 'passenger', 'passion', 'past', 'patch', 'path', 'patient',
            'pattern', 'pause', 'pay', 'payment', 'PC', 'peace', 'peak', 'peer', 'penalty', 'people', 'pepper', 'per',
            'perceive', 'percentage', 'perception', 'perfect', 'perfectly', 'perform', 'performance',
            'perhaps', 'period', 'permanent', 'permission', 'permit', 'person', 'personal', 'personality', 'personally',
            'personnel', 'perspective', 'persuade', 'pet', 'phase', 'phenomenon', 'philosophy', 'phone',
            'photo', 'photograph', 'photographer', 'phrase', 'physical', 'physically', 'physician', 'piano', 'pick',
            'picture', 'pie', 'piece', 'pile', 'pilot', 'pine', 'pink', 'pipe', 'pitch', 'place', 'plan',
            'plane', 'planet', 'planning', 'plant', 'plastic', 'plate', 'platform', 'play', 'player', 'please',
            'pleasure', 'plenty', 'plot', 'plus', 'PM', 'pocket', 'poem', 'poet', 'poetry', 'point', 'pole',
            'police', 'policy', 'political', 'politically', 'politician', 'politics', 'poll', 'pollution', 'pool',
            'poor', 'pop', 'popular', 'population', 'porch', 'port', 'portion', 'portrait', 'portray',
            'pose', 'position', 'positive', 'possess', 'possibility', 'possible', 'possibly', 'post', 'pot', 'potato',
            'potential', 'potentially', 'pound', 'pour', 'poverty', 'powder', 'power', 'powerful',
            'practical', 'practice', 'pray', 'prayer', 'precisely', 'predict', 'prefer', 'preference', 'pregnancy',
            'pregnant', 'preparation', 'prepare', 'prescription', 'presence', 'present', 'presentation',
            'preserve', 'president', 'presidential', 'press', 'pressure', 'pretend', 'pretty', 'prevent', 'previous',
            'previously', 'price', 'pride', 'priest', 'primarily', 'primary', 'prime', 'principal', 'principle',
            'print', 'prior', 'priority', 'prison', 'prisoner', 'privacy', 'private', 'probably', 'problem',
            'procedure', 'proceed', 'process', 'produce', 'producer', 'product', 'production', 'profession',
            'professional', 'professor', 'profile', 'profit', 'program', 'progress', 'project', 'prominent', 'promise',
            'promote', 'prompt', 'proof', 'proper', 'properly', 'property', 'proportion', 'proposal',
            'propose', 'proposed', 'prosecutor', 'prospect', 'protect', 'protection', 'protein', 'protest', 'proud',
            'prove', 'provide', 'provider', 'province', 'provision', 'psychological', 'psychologist',
            'psychology', 'public', 'publication', 'publicly', 'publish', 'publisher', 'pull', 'punishment', 'purchase',
            'pure', 'purpose', 'pursue', 'push', 'put', 'qualify', 'quality', 'quarter', 'quarterback',
            'question', 'quick', 'quickly', 'quiet', 'quietly', 'quit', 'quite', 'quote', 'race', 'racial', 'radical',
            'radio', 'rail', 'rain', 'raise', 'range', 'rank', 'rapid', 'rapidly', 'rare', 'rarely',
            'rate', 'rather', 'rating', 'ratio', 'raw', 'reach', 'react', 'reaction', 'read', 'reader', 'reading',
            'ready', 'real', 'reality', 'realize', 'really', 'reason', 'reasonable', 'recall', 'receive',
            'recent', 'recently', 'recipe', 'recognition', 'recognize', 'recommend', 'recommendation', 'record',
            'recording', 'recover', 'recovery', 'recruit', 'red', 'reduce', 'reduction', 'refer', 'reference',
            'reflect', 'reflection', 'reform', 'refugee', 'refuse', 'regard', 'regarding', 'regardless', 'regime',
            'region', 'regional', 'register', 'regular', 'regularly', 'regulate', 'regulation', 'reinforce',
            'reject', 'relate', 'relation', 'relationship', 'relative', 'relatively', 'relax', 'release', 'relevant',
            'relief', 'religion', 'religious', 'rely', 'remain', 'remaining', 'remarkable', 'remember',
            'remind', 'remote', 'remove', 'repeat', 'repeatedly', 'replace', 'reply', 'report', 'reporter', 'represent',
            'representation', 'representative', 'Republican', 'reputation', 'request', 'require',
            'requirement', 'research', 'researcher', 'resemble', 'reservation', 'resident', 'resist', 'resistance',
            'resolution', 'resolve', 'resort', 'resource', 'respect', 'respond', 'respondent', 'response',
            'responsibility', 'responsible', 'rest', 'restaurant', 'restore', 'restriction', 'result', 'retain',
            'retire', 'retirement', 'return', 'reveal', 'revenue', 'review', 'revolution', 'rhythm', 'rice',
            'rich', 'rid', 'ride', 'rifle', 'right', 'ring', 'rise', 'risk', 'river', 'road', 'rock', 'role', 'roll',
            'romantic', 'roof', 'room', 'root', 'rope', 'rose', 'rough', 'roughly', 'round', 'route',
            'routine', 'row', 'rub', 'rule', 'run', 'running', 'rural', 'rush', 'Russian', 'sacred', 'sad', 'safe',
            'safety', 'sake', 'salad', 'salary', 'sale', 'sales', 'salt', 'same', 'sample', 'sanction',
            'sand', 'satellite', 'satisfaction', 'satisfy', 'sauce', 'save', 'saving', 'say', 'scale', 'scandal',
            'scared', 'scenario', 'scene', 'schedule', 'scheme', 'scholar', 'scholarship', 'school',
            'science', 'scientific', 'scientist', 'scope', 'score', 'scream', 'screen', 'script', 'sea', 'search',
            'season', 'seat', 'second', 'secret', 'secretary', 'section', 'sector', 'secure', 'security',
            'see', 'seed', 'seek', 'seem', 'segment', 'seize', 'select', 'selection', 'self', 'sell', 'Senate',
            'senator', 'send', 'senior', 'sense', 'sensitive', 'sentence', 'separate', 'sequence', 'series',
            'serious', 'seriously', 'serve', 'service', 'session', 'set', 'setting', 'settle', 'settlement', 'seven',
            'several', 'severe', 'sex', 'sexual', 'shade', 'shadow', 'shake', 'shall', 'shape', 'share',
            'sharp', 'she', 'sheet', 'shelf', 'shell', 'shelter', 'shift', 'shine', 'ship', 'shirt', 'shit', 'shock',
            'shoe', 'shoot', 'shooting', 'shop', 'shopping', 'shore', 'short', 'shortly', 'shot',
            'should', 'shoulder', 'shout', 'show', 'shower', 'shrug', 'shut', 'sick', 'side', 'sigh', 'sight', 'sign',
            'signal', 'significance', 'significant', 'significantly', 'silence', 'silent', 'silver',
            'similar', 'similarly', 'simple', 'simply', 'sin', 'since', 'sing', 'singer', 'single', 'sink', 'sir',
            'sister', 'sit', 'site', 'situation', 'six', 'size', 'ski', 'skill', 'skin', 'sky', 'slave',
            'sleep', 'slice', 'slide', 'slight', 'slightly', 'slip', 'slow', 'slowly', 'small', 'smart', 'smell',
            'smile', 'smoke', 'smooth', 'snap', 'snow', 'so', 'so-called', 'soccer', 'social', 'society',
            'soft', 'software', 'soil', 'solar', 'soldier', 'solid', 'solution', 'solve', 'some', 'somebody', 'somehow',
            'someone', 'something', 'sometimes', 'somewhat', 'somewhere', 'son', 'song', 'soon',
            'sophisticated', 'sorry', 'sort', 'soul', 'sound', 'soup', 'source', 'south', 'southern', 'Soviet', 'space',
            'Spanish', 'speak', 'speaker', 'special', 'specialist', 'species', 'specific',
            'specifically', 'speech', 'speed', 'spend', 'spending', 'spin', 'spirit', 'spiritual', 'split', 'spokesman',
            'sport', 'spot', 'spread', 'spring', 'square', 'squeeze', 'stability', 'stable',
            'staff', 'stage', 'stair', 'stake', 'stand', 'standard', 'standing', 'star', 'stare', 'start', 'state',
            'statement', 'station', 'statistics', 'status', 'stay', 'steady', 'steal', 'steel',
            'step', 'stick', 'still', 'stir', 'stock', 'stomach', 'stone', 'stop', 'storage', 'store', 'storm', 'story',
            'straight', 'strange', 'stranger', 'strategic', 'strategy', 'stream', 'street',
            'strength', 'strengthen', 'stress', 'stretch', 'strike', 'string', 'strip', 'stroke', 'strong', 'strongly',
            'structure', 'struggle', 'student', 'studio', 'study', 'stuff', 'stupid', 'style',
            'subject', 'submit', 'subsequent', 'substance', 'substantial', 'succeed', 'success', 'successful',
            'successfully', 'such', 'sudden', 'suddenly', 'sue', 'suffer', 'sufficient', 'sugar', 'suggest',
            'suggestion', 'suicide', 'suit', 'summer', 'summit', 'sun', 'super', 'supply', 'support', 'supporter',
            'suppose', 'supposed', 'Supreme', 'sure', 'surely', 'surface', 'surgery', 'surprise', 'surprised',
            'surprising', 'surprisingly', 'surround', 'survey', 'survival', 'survive', 'survivor', 'suspect', 'sustain',
            'swear', 'sweep', 'sweet', 'swim', 'swing', 'switch', 'symbol', 'symptom', 'system',
            'table', 'tablespoon', 'tactic', 'tail', 'take', 'tale', 'talent', 'talk', 'tall', 'tank', 'tap', 'tape',
            'target', 'task', 'taste', 'tax', 'taxpayer', 'tea', 'teach', 'teacher', 'teaching',
            'team', 'tear', 'teaspoon', 'technical', 'technique', 'technology', 'teen', 'teenager', 'telephone',
            'telescope', 'television', 'tell', 'temperature', 'temporary', 'ten',
            'tend', 'tendency', 'tennis', 'tension', 'tent', 'term', 'terms', 'terrible', 'territory', 'terror',
            'terrorism', 'terrorist', 'test', 'testify', 'testimony', 'testing', 'text', 'than', 'thank',
            'thanks', 'that', 'the', 'theater', 'their', 'them', 'theme', 'themselves', 'then', 'theory', 'therapy',
            'there', 'therefore', 'these', 'they', 'thick', 'thin', 'thing', 'think', 'thinking',
            'third', 'thirty', 'this', 'those', 'though', 'thought', 'thousand', 'threat', 'threaten',
            'three', 'throat', 'through', 'throughout', 'throw', 'thus', 'ticket', 'tie', 'tight', 'time', 'tiny',
            'tip', 'tire', 'tired', 'tissue', 'title', 'to', 'tobacco', 'today', 'toe', 'together',
            'tomato', 'tomorrow', 'tone', 'tongue', 'tonight', 'too', 'tool', 'tooth', 'top', 'topic', 'toss', 'total',
            'totally', 'touch', 'tough', 'tour', 'tourist', 'tournament', 'toward', 'towards',
            'tower', 'town', 'toy', 'trace', 'track', 'trade', 'tradition',
            'traditional', 'traffic', 'tragedy', 'trail', 'train', 'training', 'transfer', 'transform',
            'transformation', 'transition', 'translate', 'transportation', 'travel', 'treat', 'treatment',
            'treaty', 'tree', 'tremendous', 'trend', 'trial', 'tribe', 'trick', 'trip', 'troop', 'trouble', 'truck',
            'true', 'truly', 'trust', 'truth', 'try', 'tube', 'tunnel', 'turn', 'TV',
            'twelve', 'twenty', 'twice', 'twin', 'two', 'type', 'typical', 'typically', 'ugly', 'ultimate',
            'ultimately', 'unable', 'uncle', 'under', 'undergo', 'understand',
            'understanding', 'unfortunately', 'uniform', 'union', 'unique', 'unit', 'United', 'universal', 'universe',
            'university', 'unknown', 'unless', 'unlike', 'unlikely',
            'until', 'unusual', 'up', 'upon', 'upper', 'urban', 'urge', 'us', 'use', 'used', 'useful', 'user', 'usual',
            'usually', 'utility', 'vacation', 'valley', 'valuable',
            'value', 'variable', 'variation', 'variety', 'various', 'vary', 'vast', 'vegetable', 'vehicle', 'venture',
            'version', 'versus', 'very', 'vessel', 'veteran', 'via', 'victim', 'victory', 'video', 'view', 'viewer',
            'village', 'violate', 'violation', 'violence', 'violent', 'virtually', 'virtue', 'virus', 'visible',
            'vision', 'visit', 'visitor', 'visual', 'vital', 'voice', 'volume', 'volunteer', 'vote', 'voter', 'vs',
            'vulnerable',
            'wage', 'wait', 'wake', 'walk', 'wall', 'wander', 'want', 'war', 'warm', 'warn', 'warning', 'wash', 'waste',
            'watch', 'water', 'wave', 'way', 'we', 'weak', 'wealth', 'wealthy', 'weapon', 'wear', 'weather',
            'wedding', 'week', 'weekend', 'weekly', 'weigh', 'weight', 'welcome', 'welfare', 'well', 'west', 'western',
            'wet', 'what', 'whatever', 'wheel', 'when', 'whenever', 'where', 'whereas', 'whether', 'which', 'while',
            'whisper', 'white', 'who', 'whole', 'whom', 'whose', 'why', 'wide', 'widely', 'widespread', 'wife', 'wild',
            'will', 'willing', 'win', 'wind', 'window', 'wine', 'wing', 'winner', 'winter', 'wipe', 'wire', 'wisdom',
            'wise', 'wish', 'with', 'withdraw', 'within', 'without', 'witness', 'woman', 'wonder', 'wonderful', 'wood',
            'wooden', 'word', 'work', 'worker', 'working', 'works', 'workshop', 'world', 'worried', 'worry', 'worth',
            'would', 'wound', 'wrap', 'write', 'writer', 'writing', 'wrong', 'yard', 'yeah', 'year', 'yell', 'yellow',
            'yes', 'yesterday', 'yet', 'yield', 'you', 'young', 'your', 'yours', 'yourself', 'youth', 'zone']


def flatten_list(t):   # single level of nesting removed
    return [item for sublist in t for item in sublist]


#def intersection(l1, l2):
#    lst3 = value, cat for value,cat in l1 if value,cat in l2
#    return lst3
#z = intersection([1,2,8,4], [4,1, 2])
#print("intersection", z)


def apply_WN_filter_to_list(lis):
    """ Takes a list of S2V midpoints [[real, words]...]. Returns only words that are also in WordNet   """
    if lis == []:
        return []
    else:
        dist = lis[0][0]
        wrd, PoS_tag = lis[0][1].split("|")
        a = wnl.lemmatize(wrd)
        if a in words.words():
            return [[dist, a+"|"+PoS_tag]] + apply_WN_filter_to_list(lis[1:])    # [a, PoS_tag].append ?
        else:
            return apply_WN_filter_to_list(lis[1:])


def return_best_candidates(w1, w2, PoS_tag, quantity=10):
    """ Return neighbours of w1, sorted by total similarity to w1 & w2ideally equidistant from w1 & w2.
    Remove hyphenated word_pairs.  """
    candidates_from_embeddings = s2v.most_similar(w1+"|"+PoS_tag, quantity)  # reduced similarity precision
    single_word_candidates=[]
    for cand in candidates_from_embeddings:
        if not re.search('.*[_-].*', cand[0]):  # remove word_chains
            tmp = s2v.similarity(w1+"|"+PoS_tag, cand[0])
            single_word_candidates.append([cand[0], tmp])
    #print("SWC", single_word_candidates)
    # 3000 most common english words
    candidates1_list = []
    candidates1_list2 = []
    for (x,y) in single_word_candidates:
        tmp = s2v.similarity([x], [w2 + "|" + PoS_tag])
        scr = y + tmp - abs(y-tmp) #(abs(y-tmp)**2) # but abs << 1.0
        candidates1_list.append([ scr, x ])
        # scr2 = y + tmp - (abs(y-tmp)**2)
        # candidates1_list2.append([scr2, x])
    res = sorted(candidates1_list, key=lambda x: x[0], reverse=True)
    # res2 = sorted(candidates1_list2, key=lambda x: x[0], reverse=True)
    # print(res,"\n",res2, "\n")
    return res
# return_best_candidates("cat", "dog", "NOUN")
# return_best_candidates("dog", "cat", "NOUN")
# return_best_candidates("cat", "moose", "NOUN")
# return_best_candidates("walk", "fly", "VERB")
# return_best_candidates("walk", "eat", "VERB")
# stop()

def s2v_midpoint_between_words(w1, w2, PoS_tag, quantity=10):
    query1, query2 = w1, w2
    assert query1+"|"+PoS_tag in s2v
    assert query2+"|"+PoS_tag in s2v
    count, previous_best = 0, sys.maxsize
    w1_to_w2 = sorted(return_best_candidates(w1, w2, PoS_tag, quantity), key=lambda x: x[1])
    w2_to_w1 = sorted(return_best_candidates(w2, w1, PoS_tag, quantity), key=lambda x: x[1])
    candidate_midpoints = sorted(w1_to_w2 + w2_to_w1, key = lambda x: x[0], reverse=True)
    midpoints_no_duplicates = []
    for x in range(len(candidate_midpoints)-1):   # remove duplicates
        if candidate_midpoints[x][1] == candidate_midpoints[x+1][1] or \
                candidate_midpoints[x][1].split("|")[0] == w1 or candidate_midpoints[x][1].split("|")[0] == w2:
            continue
        else:
            midpoints_no_duplicates += [candidate_midpoints[x]]
    if len(candidate_midpoints) >2 and candidate_midpoints[-1] != candidate_midpoints[-2]: #possibly add the last element
        midpoints_no_duplicates += [candidate_midpoints[-1]]
    return midpoints_no_duplicates
print("S2v midpoint between dog and cat =", s2v_midpoint_between_words('dog','cat', 'NOUN'))



def s2v_midpoint_between_words_2(w1, w2, PoS_tag):
    print("MID ", w1, w2, PoS_tag, end=" :: ")
    query1 = w1
    query2 = w2
    assert query1 + "|" + PoS_tag in s2v
    assert query2 + "|" + PoS_tag in s2v
    count = 0
    score1 = []
    score2 = []
    previous_best = sys.maxsize
    total_distances_a, total_distances_b = [], []
    while count < 2:
        neighbrs_of_a = return_best_candidates(query1, w1, PoS_tag)
        for x,y in neighbrs_of_a:
            next_dist = s2v.similarity(w1+"|"+PoS_tag, y)
            total_distances_a += [[y, x+next_dist]]
        if w1+"|"+PoS_tag in total_distances_a:
            total_distances_a.remove([w1+"|"+PoS_tag])
        neighbrs_of_b = return_best_candidates(query2, w2, PoS_tag)
        for x,y in neighbrs_of_b:
            next_dist = s2v.similarity(w2+"|"+PoS_tag, y)
            total_distances_b += [[y, x+next_dist]]
        candidates = sorted(total_distances_a + total_distances_b, key=lambda x: x[0])
        temp_d = s2v.similarity(w1+"|"+PoS_tag, w2)
        candidates.remove([w1 + "|" + PoS_tag, _])

        midpoint1_estimate = sorted(score1, key=lambda x: abs(x[0]))  # [::-1]
        midpoint2_estimate = sorted(score2, key=lambda x: abs(x[0]))  # [::-1]
        big_list = sorted(midpoint1_estimate + midpoint2_estimate, key=lambda x: abs(x[0]))
        if big_list[0][0] < previous_best:
            previous_best = big_list[0][0]
            best_answer_so_far = big_list[0][1]
        query1 = midpoint1_estimate[0][1].split("|")[0]
        query2 = midpoint2_estimate[0][1].split("|")[0]
        count += 1
        print("LOOP: ", best_answer_so_far, previous_best, end="     ")
    rslt1 = sorted(big_list, key=lambda x: abs(x[0]))
    print()
    print("ClearnResult ", end="")
    for (a, b) in rslt1:
        if not "_" in b:
            print("{: <20} {:.3f} ".format(b, a), end="    ")
    return rslt1
print()
print("S2v Midpoint_2 dog cat", s2v_midpoint_between_words_2('dog', 'cat', 'NOUN'))

#stop()

def generalise_word_UNUSED(word, pos):
    # words.words('en-basic')  # 850 words
    if pos == "NOUN":
        root = wn.morphy(word, wn.NOUN)  # + "|" + pos + '|01'
    elif pos == "VERB":
        root = wn.morphy(word, wn.VERB)  # + "|" + pos + '|01'
    if wn.synsets(root):
        my_synset = wn.synsets(root)[0]
        parents = my_synset.hypernyms()
        print(" rcrs_ovr_hypnm", end=" ")
        res = [root, my_synset] + return_hypernym_list_with_levels(my_synset)  # [:-1]
        # all_hypernyms = list(set([w for s in my_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
        #print(res, "\n \n")
print(" Gereralis word UNUS bank NOUN =", generalise_word_UNUSED("bank", "NOUN"))
print(" Gereralis word UNUS bank VERB=", generalise_word_UNUSED("bank", "VERB"))


global sy_set_res
sy_set_res = []

def recurse_over_hypernymns(sy_set): # synonym_set
    if sy_set.lemmas()[0].name() == "entity" or sy_set.lemmas()[0].name() == "placental":
        return  # []
    else:
        parent1 = [sy_set.hypernyms()[0]]
        # print(parent1, end="  ")
        sy_set_res.extend([parent1])
        return recurse_over_hypernymns(parent1[0])


def add_levels_to_hypernym_list(elem_string, maxy):
    res = []
    elem_string.reverse()
    for indx, val in enumerate(elem_string):
        res.append([val, indx])
    res.reverse()
    return res


def add_levels_to_hypernym_tuples(elem_string, maxy):
    res = []
    elem_string.reverse()
    for indx, val in enumerate(elem_string):
        res.append((val, indx))
    res.reverse()
    return res


def return_hypernym_list_with_levels(syn_set, elem_string):
    """ Returns an inverted list of Hypernyms with WordNet levels - integers representing semantic depth"""
    branch = []
    if syn_set.lemmas()[0].name() in ["entity", "touch"]:     # Synset.root_hypernyms()
        res = add_levels_to_hypernym_list(elem_string, len(elem_string))
        return res
    else:
        parents = syn_set.hypernyms()
        if parents == []:   # dod 8 April 2022
            return elem_string
        parent1 = parents[0]                  # what if its a DAG?
        #if len(parents) > 1:
        #    branch = return_hypernym_list_with_levels(parents[1], elem_string)
        elem_string.append(parent1.lemmas()[0].name())
        return return_hypernym_list_with_levels(parent1, elem_string) + branch
#print("Print hypernym DAG", return_hypernym_list_with_levels(wn.synsets('dog', wn.NOUN)[0], []))



def return_hypernym_tuples_with_levels(syn_set, elem_string):
    """ Hypernyms with WordNet levels - integers representing semantic depth"""
    if syn_set.lemmas()[0].name() in ["entity", "touch"]:     # Synset.root_hypernyms()
        res = add_levels_to_hypernym_tuples(elem_string, len(elem_string))
        return res
    else:
        parents = syn_set.hypernyms()
        if parents == []:   # dod 8 April 2022
            return elem_string
        parent1 = parents[0]
        elem_string.append(parent1.lemmas()[0].name())
        return return_hypernym_tuples_with_levels(parent1, elem_string)


def flatten_list(t):   # single level of nesting removed
    return [item for sublist in t for item in sublist]


def remove_duplicates(lis): # list of sublist pairs
    """ from a list of sub-list pairs [ [word, level], ...], sorted on 2 attributes with adjacent duplicates."""
    singleton_list = []
    for x in range(len(lis)-1):
        if lis[x][0] != lis[x+1][0]:
            singleton_list += [lis[x]]
    if len(lis)> 2 and singleton_list[-1][0] != lis[-1][0]:
        singleton_list += [lis[-1]]
    return singleton_list
#remove_duplicates([ ['dog', 6], ['sausage', 6], ['device', 6], ['device', 6], ['organism', 5], ['unpleasant_person', 5], ['chap', 5], ['villain', 5], ['meat', 5], ['instrumentality', 5], ['instrumentality', 5], ['living_thing', 4], ['unwelcome_person', 4], ['male', 4], ['unwelcome_person', 4], ['food', 4], ['artifact', 4], ['artifact', 4], ['whole', 3], ['person', 3], ['person', 3], ['person', 3], ['solid', 3], ['whole', 3], ['whole', 3], ['object', 2], ['causal_agent', 2], ['causal_agent', 2], ['causal_agent', 2], ['matter', 2], ['object', 2], ['object', 2], ['physical_entity', 1], ['physical_entity', 1], ['physical_entity', 1], ['physical_entity', 1], ['physical_entity', 1], ['physical_entity', 1], ['physical_entity', 1], ['entity', 0], ['entity', 0], ['entity', 0], ['entity', 0], ['entity', 0], ['entity', 0], ['entity', 0]])


def merge_superordinate_categories_by_level(list_o_lists):
    if list_o_lists == []:
        return
    res = flatten_list(list_o_lists)
    res2 = sorted(res, key=lambda x: (x[1], x[0]), reverse = True)
    res3 = remove_duplicates(res2)
    return res3
#tmp =[[['dog', 13], ['canine', 12], ['carnivore', 11], ['placental', 10], ['mammal', 9], ['vertebrate', 8], ['chordate', 7], ['animal', 6], ['organism', 5], ['living_thing', 4], ['whole', 3], ['object', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 7], ['unpleasant_woman', 6], ['unpleasant_person', 5], ['unwelcome_person', 4], ['person', 3], ['causal_agent', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 6], ['chap', 5], ['male', 4], ['person', 3], ['causal_agent', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 6], ['villain', 5], ['unwelcome_person', 4], ['person', 3], ['causal_agent', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 7], ['sausage', 6], ['meat', 5], ['food', 4], ['solid', 3], ['matter', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 9], ['catch', 8], ['restraint', 7], ['device', 6], ['instrumentality', 5], ['artifact', 4], ['whole', 3], ['object', 2], ['physical_entity', 1], ['entity', 0]], [['dog', 8], ['support', 7], ['device', 6], ['instrumentality', 5], ['artifact', 4], ['whole', 3], ['object', 2], ['physical_entity', 1], ['entity', 0]]]
#merge_superordinate_categories_by_level(tmp)


def return_first_WN_super_category_list(in_word, pos): # WordNet
    """ returns reversed list of most-likely hypernyms for input word  """
    if pos == "NOUN":
        synset_list = wn.synsets(in_word, wn.NOUN)
    elif pos == "VERB":
        synset_list = wn.synsets(in_word, wn.VERB)
    if len(synset_list) == 0:
        return []
    res = []
    if in_word == "Saturn":
        xxx = 0
    t = return_hypernym_list_with_levels(synset_list[0], list())
    if t != [] and isinstance(t[0], list):
        t.insert(0, [in_word, t[0][1] + 1])
        res.append(t)
    elif len(t) > 1:
        print(" **DAG node", in_word, t[1], end="*** ")
        res = t[:1]
    else:
        res = t
    return res #hypernyms
#print("return_first_WN_super_category_list", return_first_WN_super_category_list('dog', 'NOUN'))
#print("return_first_WN_super_category_list", return_first_WN_super_category_list('run', 'VERB'))
#z = return_first_WN_super_category_list('cow', 'NOUN')
#print("return_first_WN_super_category_list", z, z[0][0][1])


def return_all_super_categories_list(in_word, pos): # WordNet
    if pos == "NOUN":
        synset_list = wn.synsets(in_word, wn.NOUN)  # + "|" + pos + '|01'
    elif pos == "VERB":
        synset_list = wn.synsets(in_word, wn.VERB)  # + "|" + pos + '|01'
    #synset_list = wn.synsets(in_word)
    if len(synset_list) == 0:
        return []
    res = []
    for z in synset_list:
        t = return_hypernym_list_with_levels(z, list())
        if t != [] and isinstance(t[0], list):
            t.insert(0, [in_word, t[0][1] + 1])
            res.append(t)
        elif len(t) >1:
            #print("Multiple parents", end=" ")  # skip verbs for now
            res.append(t)
        else:
            res.append(t)
    res2 = merge_superordinate_categories_by_level(res)
    return res2
#print("\nBBBB", return_all_super_categories_list('run', 'VERB'))


def find_super_category_tuples_DEPRECATED(in_word): # WordNet
    synset_list = wn.synsets(in_word)
    if len(synset_list) == 0:
        return []
    res = []
    for z in synset_list:
        t = return_hypernym_list_with_levels(z, list())
        if isinstance(t[0], list):
            t.insert(0, [in_word, t[0][1] + 1])
            res.append(t)  # (return_hypernym_list_with_levels(z, list()))
        else:
            print("v", end=" ")  # skip verbs for now
    return res #hypernyms
#print("DDDDD", find_super_category_tuples("dog"))


def magic_function(s2v_similarity, n):
    print(s2v_similarity, n)
    return n*(s2v_similarity ** 2)

###############################################################################################################
###############################################################################################################
###############################################################################################################

def simplifyLCS(synsetName):  # LCS - Lowest Common Subsumer
    """ Simplify a synset to just the synset name.
    It accepts either an isolated synset name of a flat list of synsets.
    simplifyLCS(["Synset('object.n.01')"]) -> "object"""
    if isinstance(synsetName, list):
        synsetName = simplifyLCS(synsetName[0]) + simplifyLCS(synsetName[1:])
    elif (isinstance(synsetName, str)) and ("Synset" in synsetName):
        y = synsetName.find('(') + 2
        z = synsetName.find('.') + 5
        synsetName = synsetName[y:z].replace('.', '(', 1) + ")"
    elif (isinstance(synsetName, list)) and (len(synsetName) > 1):
        simplifyLCS(synsetName[0]).append(simplifyLCS(synsetName[1:]))
        simplifyLCS(str(synsetName[0])).append(simplifyLCSList(synsetName[1:]))  # 11/10
    elif str(synsetName)[:6] == "Synset":  # instance of <class 'nltk.corpus.reader.wordnet.Synset'>
        ssString = str(synsetName)
        y = ssString.find('(') + 2
        z = ssString.find('.') + 5
        synsetName = ssString[y:z].replace('.', '(', 1) + ")"
    return synsetName  # [synsetName]


# simplifyLCSList("[Synset('whole.n.02')]")

def simplifyLCSList(synsetList):
    if (synsetList is None):
        return "none1"
    elif (synsetList == []):
        return ""
    elif synsetList == "none":
        return "none"
    elif (isinstance(synsetList, str)):
        z = simplifyLCS(synsetList)
        return z
    elif (isinstance(synsetList, list)) and (len(synsetList) > 1):
        return str(simplifyLCS(synsetList[0])) + "_" + str(simplifyLCSList(synsetList[1:]))
    elif (isinstance(synsetList, list)) and (len(synsetList) == 1):
        return simplifyLCS(synsetList[0])
    else:
        print(" sLCSL5", end="")
        zz = simplifyLCS(synsetList)
        return zz

# semcor_ic = wordnet_ic.ic('ic-semcor.dat')
brown_ic = wordnet_ic.ic('ic-brown.dat')

def return_level_of_word(in_word, pos):
    """ return the integer depth and superordinate synsets for a given word. Shallowest level used. """
    if pos == "NOUN":
        synset_list = wn.synsets(in_word, wn.NOUN)[0].hypernym_paths()
    elif pos == "VERB":
        synset_list = wn.synsets(in_word, wn.VERB)[0].hypernym_paths()
    if synset_list == []:
        return 0
    lowest_level, superordinate_synsets = len(synset_list[0]), synset_list[0][-1]
    for liz in synset_list[1:]:
        if len(liz) < lowest_level:
            lowest_level = len(liz)
            superordinate_synsets = liz[-1]
    return lowest_level, superordinate_synsets
#print(" return_level_of_word dog Noun", return_level_of_word("dog", "NOUN"))


def wn_sim_mine(w1, w2, partoS, use_lexname=True):
    """ wn_sim_mine("create","construct", 'v') -> [0.6139..., 'make(v.03)', 0.666..., 'make(v.03)']
     Originally written for Map2Graphs.py. """
    global LCSLlist
    lexname_out = ""
    lin_max = wup_max = 0
    LCSL_temp = LCSW_temp = []
    flag = False
    w1 = w1.lower()
    w2 = w2.lower()
    if partoS == "NOUN":
        partoS = 'n'
    elif partoS == "VERB":
        partoS = 'v'
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    if wn1 is not None:
        w1 = wn1
    if wn2 is not None:
        w2 = wn2
    LCSL = LCSW = "-"
    if w1 == w2:
        return [1, w1, 1, w2]
    else:
        syns1 = wn.synsets(w1, pos=partoS)
        syns2 = wn.synsets(w2, pos=partoS)
        for ss1 in syns1:
            if use_lexname:
                ss1_lex_nm = ss1.lexname()
            for ss2 in syns2:
                if use_lexname:
                    ss2_lex_nm = ss2.lexname()
                lin = ss1.lin_similarity(ss2, brown_ic)  # semcor_ic)  # brown_ic
                wup = ss1.wup_similarity(ss2)
                if lin > lin_max:
                    lin_max = lin
                    ss1_Lin_temp = ss1
                    ss2_Lin_temp = ss2
                    LCSL_temp = ss1.lowest_common_hypernyms(ss2)
                if wup > wup_max:
                    wup_max = wup
                    ss1_Wup_temp = ss1
                    ss2_Wup_temp = ss2
                    LCSW_temp = ss1.lowest_common_hypernyms(ss2)  # may return []
                if lin is None:
                    lin = 0
                if wup is None:
                    wup = 0
                if use_lexname and flag ==  False:
                    if ss1_lex_nm == ss2_lex_nm and flag == False:
                        lexname_out = ss1_lex_nm
                        flag = True
        if lin_max > 0:
            LCSL_temp = ss1_Lin_temp.lowest_common_hypernyms(ss2_Lin_temp)
            LCSL = simplifyLCSList(LCSL_temp)
        if wup_max > 0:
            LCSW_temp = ss1_Wup_temp.lowest_common_hypernyms(ss2_Wup_temp)
            LCSW = simplifyLCSList(LCSW_temp)
        if lin_max < 0.0000000001:
            lin_max = 0
            LCSW = simplifyLCSList(LCSW_temp)
        # print(" &&&& LCSW_temp2 ", LCSW_temp)
        if LCSW_temp == []:
            LCSW = "Synset('null." + partoS + ".02')"
        #write_to_wn_cache_file(w1, w2, partoS, lin_max, wup_max, LCSL, LCSW+"-"+lexname_out)
    LCSLlist.append(LCSL)  # for the GUI presentation
    if LCSL_temp == []:
        LCSL = "Synset('null." + partoS + ".0404')"
    if LCSW_temp == []:
        LCSW = "Synset('null." + partoS + ".0403')"
    return [lin_max, LCSL, wup_max, LCSW, lexname_out]
    # return [lin_max, LCSL, wup_max, LCSW, lexname_out]
print("wn_sim_mine(create,construct, VERB) =", wn_sim_mine("create","construct", 'VERB'))
print("wn_sim_mine(cat, dog, NOUN) =", wn_sim_mine('cat', 'dog', "NOUN"))
print("  -  - - - - - - - ")


def wn_sim_first(w1, w2, pos):
    global LCSLlist
    lin_max = wup_max = 0
    LCSL_temp = LCSW_temp = []
    LCSL = LCSW = lexname_out = "-"
    w1 = w1.lower()
    w2 = w2.lower()
    if pos == "NOUN":
        partoS = 'n'
    elif pos == "VERB":
        partoS = 'v'
    wn1 = wn.morphy(w1, partoS)
    wn2 = wn.morphy(w2, partoS)
    if wn1 is None or wn2 is None:
        return [0, "-", 0, "-", "-"]
    elif w1 == w2:
        return [1, w1, 1, w2]
    w1 = wn1
    w2 = wn2
    ss1 = wn.synsets(w1, partoS)[0]
    ss2 = wn.synsets(w2, partoS)[0]
    lin = ss1.lin_similarity(ss2, brown_ic)  # semcor_ic)  # brown_ic
    wup = ss1.wup_similarity(ss2)
    LCS = ss1.lowest_common_hypernyms(ss2)[0]
    w1_level, w1_supers = return_level_of_word(w1, pos)
    w2_level, w2_supers = return_level_of_word(w2, pos)
    ss1_lex_nm = ss1.lexname()
    ss2_lex_nm = ss2.lexname()
    return w1_level, w2_level, lin, wup, LCS, ss1_lex_nm, ss2_lex_nm



def generalise_word_pair(w1, w2, pos=True):
    """ find generic term between 2 words    """
    super_1a = return_first_WN_super_category_list(w1, pos)[0]
    super_2a = return_first_WN_super_category_list(w2, pos)[0]
    level_w1 = super_1a[0][1]  # level in WordNet hierarchy
    level_w2 = super_2a[0][1]
    min_WN_level = min(level_w1, level_w2)
    super_1 = super_1a[-min_WN_level:]
    super_2 = super_2a[-min_WN_level:]
    both_supers = sorted(super_1 + super_2, key=lambda x: x[1], reverse = True)
    supers_candidates = remove_duplicates(both_supers)
    t1 = s2v[w1 + "|" + pos]
    t2 = s2v[w1 + "|" + pos]
    if t1 is None or t2 is None:
        return None
    w1_to_w2_distance = s2v.similarity(w1+"|"+pos, w2+"|"+pos)
    print("S2v", w1, " <-> ", w2, pos, "   ", w1_to_w2_distance)
    print("::", super_1a[:4], "  --  ", super_2a[:4], min_WN_level)
    midpoints_list = s2v_midpoint_between_words(w1, w2, pos)
    print("MiddleMost   :", len(midpoints_list), str(midpoints_list[0][1]))
    midpoints_in_wn = []
    for x in midpoints_list:     # find a midpoint word  thats also in WordNet
        if x[1].split('|')[0] in wn_lemmas:
            midpoints_in_wn += [x[1].split('|')]
    print("S2v midpoint candidates", midpoints_in_wn)
    super_categories = return_all_super_categories_list(midpoints_list[0][1].split('|')[0], pos)[0][-min_WN_level:] # focus only on middlemost word
    n= len(super_categories)
    print("S2v Middle:", midpoints_list[0] , "  ", str(super_categories), end=" ")
    index_narrowest_cat = magic_function(w1_to_w2_distance, n)  # linear?
    # print("#K:", index_narrowest_cat, end=" ")
    integerized_index = n-round(index_narrowest_cat)
    print("INDX ", n, w1_to_w2_distance, ";;", integerized_index, index_narrowest_cat)
    if integerized_index >= n:
        integerized_index = n-1
    if integerized_index >= 0:
        print(w1, w2, pos, " -> " + str(super_categories[integerized_index]), index_narrowest_cat)
    else:
        print(w1, w2, pos, "#GENERALISATION: none")


def wordnetify(w, pos):
    w_wn = "NONE"
    if not w in wn_lemmas:
        if pos == "verb":
            w_wn = wn.morphy(w, wn.VERB)
        elif pos == "noun":
            w_wn = wn.morphy(w, wn.NOUN)
    xxx=0
    return w_wn
print("wordnetify denied verb", wordnetify("denied", "verb"))


def generalise_word_pair_quick(w1, w2, pos=True):
    """ Find a suitable superordinate term connecting 2 given words. """
    super_1 = return_first_WN_super_category_list(w1, pos)
    super_2 = return_first_WN_super_category_list(w2, pos)
    level_w1 = super_1[0][1]  # level in WordNet hierarchy
    level_w2 = super_2[0][1]
    min_win_level = max(level_w1, level_w2)
    t1 = s2v[w1 + "|" + pos]
    t2 = s2v[w1 + "|" + pos]
    if t1 is None or t2 is None:
        print("Unknown word/s")
        return None
    w1_to_w2_similarity = s2v.similarity(w1+"|"+pos, w2+"|"+pos)
    print("   ",w1, " <--",w1_to_w2_similarity,"--> ", w2, pos, end="  ")
    midpoints = s2v_midpoint_between_words(w1, w2, pos)
    print("\nMiddleMost      :" + str(midpoints) )
    cleaned_candidate_list = apply_WN_filter_to_list(midpoints)
    print("Cleaned Candidates:" + str(cleaned_candidate_list) )
    print()
    super_categories_w1 = return_all_super_categories_list(w1, pos)
    super_categories_w2 = return_all_super_categories_list(w2, pos)
    print("::", super_categories_w1, "   ---   ", super_categories_w2)
    super_categories = return_all_super_categories_list(midpoints[0][1].split('|')[0]) # focus only on middlemost word
    # maybe add midpoints to the super_categories
    n= len(super_categories)
    print("#SUPER CATEGORIES:", midpoints[0], "  ", str(super_categories), end=" ")

    index_narrowest_cat = magic_function(w1_to_w2_similarity, n)  # linear?
    # print("#K:", index_narrowest_cat, end=" ")
    integerized_index = n-round(index_narrowest_cat)
    print("INDX ", n, w1_to_w2_similarity, ">", (w1_to_w2_similarity ** 2), ";;", integerized_index, index_narrowest_cat)
    if integerized_index >= n:
        integerized_index = n-1
    if integerized_index >= 0:
        print(w1, w2, pos, " -> " + str(super_categories[integerized_index]), index_narrowest_cat)
    else:
        print(w1, w2, pos, "#GENERALISATION: none")


word_pairs = [ ("house", "boat", "NOUN"), ("road", "river", "NOUN"), ("walk", "run", "VERB"), ("dog", "cat", "NOUN"),
               ("tumor", "fortress", "NOUN"),  ("accept", "say", "VERB")]
#                  ("dog", "tiger", "NOUN"),  ("wolf", "tiger", "NOUN"), ("dog", "cow", "NOUN"), ("dog", "tree", "NOUN"),
#                  ("desk", "chair", "NOUN"), ("Tuesday", "Saturn", "NOUN"), ("bird", "wing", "NOUN"),
#                  ("drive", "fly", "VERB"),  ("worker", "roads", "NOUN") ]

def average_words_experiment(examples):
    for i in examples:
        #print(*i)
        print(" --- QUICK --- ")
        #generalise_word_pair_quick(*i)
        generalise_word_pair(*i)
        print("=======================================================================")


w0 = wordnet.synset('tumor.n.01')
w1 = wordnet.synset('fortress.n.01')
z0 = w0._shortest_hypernym_paths(w1)
z1 = w1._shortest_hypernym_paths(w0)
z2 = w0.shortest_path_distance(w1)
x = wordnet.synset('accept.v.01')
y = wordnet.synset('say.v.01')
z5 =x._shortest_hypernym_paths(y)
z6 =x.shortest_path_distance(y)
#print(z6)

if __name__=="__main__":
    average_words_experiment(word_pairs)

# see Map2Graphs.wn_sim_mine(w1,w2,pos)

# l = ('a' ('b', 'c') ('d', ('e', ('f')), g)