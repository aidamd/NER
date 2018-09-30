from __future__ import print_function, division

import sys
sys.path.insert(0, '..')
from utils import conlleval


def read_conll(fname):
  """Read data from any files with fixed format.
  Each line of file should be a space-separated token information,
  in which information starts from the token itself.
  Each sentence is separated by a empty line.

  e.g. 'Apple I-NP I-NP I-ORG' could be one line

  Args:
      fname (str): file path for reading data.

  Returns:
      sentences (list):
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label]   
  """
  sentences, prev_sentence = [], []
  with open(fname) as f:
    for line in f:
      if not line.strip():
        if prev_sentence:
          sentences.append(prev_sentence)
        prev_sentence = []
        continue
      prev_sentence.append(list(line.strip().split()))
  return sentences


def conll_to_data_stream(sentences, write_to_file=''):
  """Convert conll data to a string of data stream and ready to write

  Args:
      sentences (list): A list of sentences to output
      write_to_file (str, optional):
        If a file path, write data stream to that file

  Returns:
      data_stream: A list of strings, each line indicating a line in file
  """
  data_stream = []
  for sentence in sentences:
    for tup in sentence:
      data_stream.append(' '.join(tup))
    data_stream.append('')
  if write_to_file:
    with open(write_to_file, 'w') as f:
      f.write('\n'.join(data_stream))
  return data_stream


def add_column(sentences, columns):
  """Add column to list of sentences
  
  Args:
      sentences (list): 
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label] 
      columns (list): Same format as sentences, but only one feature presented in
        token information.
  
  Returns:
      new_sentences: same format as sentences
  """
  new_sentences = []
  for sentence, column in zip(sentences, columns):
    new_sentences.append(
        [tup + [col] for tup, col in zip(sentence, column)]
    )
  return new_sentences


def get_column(sentences, i):
  """Get a column of information from sentences
  
  Args:
      sentences (list): 
        Sentences is a list of sentences.
        Sentence is a list of token information.
        Token information is in format: [token, feature_1, ..., feature_n, tag_label] 

      i (int): A index to retrieve from. Can be positive or negative (backward)
  
  Returns:
      columns (list): Same format as sentences, but only one feature presented in
        token information.
  """
  columns = []
  for sentence in sentences:
    columns.append([tup[i] for tup in sentence])
  return columns


def tags_from_conll(tags, scheme='bio'):
  """Convert a tag sequence from conll format to other format.

  Args:
      tags (list): A list of tag sequence OR a list of tags
        tag is either 'O', 'B-TYPE' or 'I-TYPE'
      scheme (str, optional): A tagging scheme
        Either 'bio', 'bioe' or 'bioes'

  Returns:
      new_tags: same format as tags
  """
  def entity_span_from_conll(entity_span, scheme=scheme):
    if not entity_span:
      return entity_span
    # Logic are performed in order of precedence.
    if 'e' in scheme:
      entity_span[-1] = 'E' + entity_span[-1][1:]
    if 'b' in scheme:
      entity_span[0] = 'B' + entity_span[0][1:]
    if 's' in scheme and len(entity_span) == 1:
      entity_span[0] = 'S' + entity_span[0][1:]
    if 'i' in scheme:
      for i in range(1, len(entity_span) - 1):
        entity_span[i] = 'I' + entity_span[i][1:]
    return entity_span

  new_tags = tags[:]
  if not new_tags:
    return new_tags
  if isinstance(tags[0], str):
    new_tags = [new_tags]

  for k, sent_tag in enumerate(new_tags):
    i = 0
    for j, tag in enumerate(sent_tag):
      flag = False
      if tag[0] in 'BO':  # 'O' and 'B' indicates the end of previous sequence
        flag = True
      # If two tags are different, 'I' is also an indicator of separation
      elif tag[0] == 'I' and j and sent_tag[j - 1][1:] != tag[1:]:
        flag = True
      if flag:
        sent_tag[i:j] = entity_span_from_conll(sent_tag[i:j], scheme=scheme)
        i = j + (tag[0] == 'O')  # If tag is not 'O', we should include it in following sequence
        continue
    sent_tag[i:] = entity_span_from_conll(sent_tag[i:], scheme=scheme)

  if isinstance(tags[0], str):
    new_tags = new_tags[0]
  return new_tags


def data_from_conll(sentences, scheme='bio'):
  """Convert sentences in conll format to required format.
  Can also be used to convert a sequence of tags to required format.

  Args:
      sentences (list): A list of sentences
      scheme (str, optional): A tagging scheme
        Either 'bio', 'bioe' or 'bioes'

  Returns:
      new_sentences: A list of sentences with tags in required format
  """
  new_sentences = []
  for sentence in sentences:
    tags = [tup[-1] for tup in sentence]
    new_tags = tags_from_conll(tags)
    new_sentences.append([
        tup[:-1] + [tag] for tup, tag in zip(sentence, new_tags)
    ])
  return new_sentences


def tags_to_conll(tags):
  """Convert a tag sequence to conll format from our format.

  Args:
      tags (list): A list of tag sequence OR a list of tags
        tag is either 'B/I/E/S-TYPE' or 'O'

  Returns:
      new_tags: same format as tags
  """
  def entity_span_to_conll(entity_span, prev_is_same_entity=False):
    if not entity_span:
      return entity_span
    for i in range(len(entity_span)):
      entity_span[i] = 'I' + entity_span[i][1:]
    if prev_is_same_entity:
      entity_span[0] = 'B' + entity_span[0][1:]
    return entity_span

  new_tags = tags[:]
  if not new_tags:
    return new_tags
  if isinstance(tags[0], str):
    new_tags = [new_tags]

  for k, sent_tag in enumerate(new_tags):
    i = 0
    for j, tag in enumerate(sent_tag):
      if tag[0] in 'OBS':
        prev_is_same_entity = i and (sent_tag[i - 1][1:] == sent_tag[i][1:])
        # print(i, j, sent_tag[i-1], sent_tag[i], sent_tag[i - 1][1:] == tag[1:])
        sent_tag[i:j] = entity_span_to_conll(sent_tag[i:j], prev_is_same_entity=prev_is_same_entity)
        i = j + (tag[0] == 'O')
      else:
        continue
    prev_is_same_entity = i and i <= j and (sent_tag[i - 1][1:] == sent_tag[i][1:])
    sent_tag[i:] = entity_span_to_conll(sent_tag[i:], prev_is_same_entity=prev_is_same_entity)

  if isinstance(tags[0], str):
    new_tags = new_tags[0]
  return new_tags


def data_to_conll(sentences):
  """Convert sentences to conll format.
  Can also be used to convert a sequence of tags to conll format.

  Args:
      sentences (list): A list of sentences

  Returns:
      new_sentences: A list of sentences with tags in conll format
  """
  new_sentences = []
  for sentence in sentences:
    tags = [tup[-1] for tup in sentence]
    new_tags = tags_to_conll(tags)
    new_sentences.append([
        tup[:-1] + [tag] for tup, tag in zip(sentence, new_tags)
    ])
  return new_sentences


# test_tags = 'I-ORG B-ORG I-ORG I-PER O I-PER I-PER I-PER'.split()
# test_tags = 'O O I-ORG I-ORG I-ORG I-MISC I-MISC I-PER I-PER I-PER'.split()
test_tags = 'I-ORG B-ORG B-ORG O'.split()


def main():
  print(tags_from_conll(test_tags, scheme='bioes'))
  print(tags_to_conll(tags_from_conll(test_tags, scheme='bioes')))

  conll_sents = read_conll('../data/eng.testa')
  print(conll_sents[1])
  conll_tags = get_column(conll_sents, -1)
  tags = tags_from_conll(conll_tags, scheme='bio')
  print(tags[1])
  new_conll_sents = add_column(conll_sents, tags_to_conll(conll_tags))
  print(new_conll_sents[1])
  conll_to_data_stream(new_conll_sents, write_to_file="tmp.testa")
  conlleval.evaluate(conll_to_data_stream(new_conll_sents))

if __name__ == '__main__':
  main()
