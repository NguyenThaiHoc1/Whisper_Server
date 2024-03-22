def reprocess(texts, src_lang):
    list_text = [texts]
    if src_lang == 'ja_XX':
        char = '。'
        indexes = [index for index, c in enumerate(texts) if c == char]
        start = 0
        list_text = []
        for index in indexes:
            end = index
            split_text = texts[start:end + 1].strip()
            start = end + 1
            if split_text:
                list_text.append(
                    split_text
                )
        list_text.append(
            texts[start:].strip()
        )

    for idx, text in enumerate(list_text):
        if text == "":
            list_text.pop(idx)

    return list_text

