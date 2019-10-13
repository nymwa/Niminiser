import re, regex
import numpy as np
import pandas as pd

# 単語を保持
class Nimis:
	nimis = ['a', 'akesi', 'ala', 'alasa', 'ale', 'ali', 'anpa', 'ante', 'anu', 'apeja', 'awen', 'e', 'en', 'esun', 'ijo', 'ike', 'ilo', 'insa', 'jaki', 'jan', 'jelo', 'jo', 'kala', 'kalama', 'kama', 'kan', 'kasi', 'ken', 'kepeken', 'kijetesantakalu', 'kili', 'kin', 'kipisi', 'kiwen', 'ko', 'kon', 'kule', 'kulupu', 'kute', 'la', 'lape', 'laso', 'lawa', 'leko', 'len', 'lete', 'li', 'lili', 'linja', 'lipu', 'loje', 'lon', 'luka', 'lukin', 'lupa', 'ma', 'majuna', 'mama', 'mani', 'meli', 'mi', 'mije', 'moku', 'moli', 'monsi', 'monsuta', 'mu', 'mun', 'musi', 'mute', 'namako', 'nanpa', 'nasa', 'nasin', 'nena', 'ni', 'nimi', 'noka', 'o', 'oko', 'olin', 'ona', 'open', 'pakala', 'pake', 'pali', 'palisa', 'pan', 'pana', 'pata', 'pi', 'pilin', 'pimeja', 'pini', 'pipi', 'poka', 'poki', 'pona', 'pu', 'sama', 'seli', 'selo', 'seme', 'sewi', 'sijelo', 'sike', 'sin', 'sina', 'sinpin', 'sitelen', 'sona', 'soweli', 'suli', 'suno', 'supa', 'suwi', 'tan', 'taso', 'tawa', 'telo', 'tenpo', 'toki', 'tomo', 'tu', 'unpa', 'uta', 'utala', 'walo', 'wan', 'waso', 'wawa', 'weka', 'wile']

# 音節を保持
class Sylls:
	sylls = [s.upper() for s in ['a', 'an', 'e', 'en', 'i', 'in', 'ja', 'jan', 'je', 'jen', 'jo', 'jon', 'ju', 'jun', 'ka', 'kan', 'ke', 'ken', 'ki', 'kin', 'ko', 'kon', 'ku', 'kun', 'la', 'lan', 'le', 'len', 'li', 'lin', 'lo', 'lon', 'lu', 'lun', 'ma', 'man', 'me', 'men', 'mi', 'min', 'mo', 'mon', 'mu', 'mun', 'na', 'nan', 'ne', 'nen', 'ni', 'nin', 'no', 'non', 'nu', 'nun', 'o', 'on', 'pa', 'pan', 'pe', 'pen', 'pi', 'pin', 'po', 'pon', 'pu', 'pun', 'sa', 'san', 'se', 'sen', 'si', 'sin', 'so', 'son', 'su', 'sun', 'ta', 'tan', 'te', 'ten', 'to', 'ton', 'tu', 'tun', 'u', 'un', 'wa', 'wan', 'we', 'wen', 'wi', 'win']]

# <Q>の判定
class Proper:
	@staticmethod
	def proper(word):
		return re.match(r'^([AOEUI]|[KSNPML][aoeui]|[TJ][aoue]|W[aei])(n(?![mnaoeui]))?(([ksnpml][aoeui]|[tj][aoeu]|w[aei])(n(?![mnaoeui]))?)*$', word)
	@staticmethod
	def syllables(word):
		return re.findall(r'(?:[KLMNPS]?[AIUEO]|[TJ][AOEU]|W[AEI])(?:N(?![MNAOEUI]))?', word.upper())

# <SYM>の判定
class Symbols:
	symbols = set([
			'^^','^.^','^_^','xD','XD',':)','^-^','(:','[:',':D',':-D','{:',':}','XD','X)','X3',':o','D:',
			'o_O','@_@',':3','>:-(','>:-A','>:-3',':_;','*q*','*Q*',':/',':(',':-(',':x','x:',
			'>.>','¬¬','=_=','-_-','≡╹ω╹≡'])
	p_emoji1 = regex.compile(r'\p{Emoji=Yes}+')
	p_emoji2 = re.compile("["
			u"\U0001F600-\U0001F64F"
			u"\U0001F300-\U0001F5FF"
			u"\U0001F680-\U0001F6FF"
			u"\U0001F1E0-\U0001F1FF"
			"]+", flags=re.UNICODE)
	@staticmethod
	def symbol(tok):
		return tok in Symbols.symbols or Symbols.p_emoji1.search(tok) or re.match(Symbols.p_emoji2, tok)

# 固有名詞，数詞を区別しない語彙クラス
class Voca:
	specs = ['<PAD>', '<BOS>', '<EOS>', '<EMP>', '<UNK>', '<SYM>']
	punct = ['!', "''", ',', '-', '.', ':', '?', '``']
	nimis = Nimis.nimis
	digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	sylls = Sylls.sylls
	toks = specs + punct + nimis + digit + sylls + ['+']
	n2t = toks
	t2n = {t:i for i,t in enumerate(toks)}
	@staticmethod
	def is_token(tok):
		return tok in Voca.specs or tok in Voca.nimis
	@staticmethod
	def word_to_id(tok):
		if tok in Voca.nimis:
			return Voca.t2n[tok]
		elif Proper.proper(tok):
			return Proper.syllables[tok]
		elif word.isnumeric():
			return Voca.w2n['<NUM>']
		elif Voca.syms.symbol(word):
			return Voca.w2n['<SYM>']
		else:
			return Voca.w2n['<UNK>']

# tokeniser
# 予めスペース区切りにしたリストを渡す
class Niminiser:
	@staticmethod
	def niminise(lst):
		flag = None
		res = []
		for elem in lst:
			if elem in Voca.specs or elem in Voca.punct or elem in Voca.nimis:
				flag = None
				res += [Voca.t2n[elem]]
			elif Proper.proper(elem):
				if flag == 'proper':
					res += [Voca.t2n['+']]
				flag = 'proper'
				res += [Voca.t2n[t] for t in Proper.syllables(elem)]
			elif re.match(r'[0-9]+', elem):
				if flag == 'number':
					res += [Voca.t2n['+']]
				flag = 'number'
				res += [Voca.t2n[n] for n in list(elem)]
			elif Symbols.symbol(elem):
				flag = None
				res += [Voca.t2n['<SYM>']]
			else:
				flag = None
				res += [Voca.t2n['<UNK>']]
		return res

