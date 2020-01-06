from presidio_evaluator import span_to_tag

BILOU_SCHEME = "BILOU"
BIO_SCHEME = "BIO"


def test_span_to_bio_multiple_tokens():
    text = "My Address is 409 Bob st. Manhattan NY. I just moved in"
    start = 14
    end = 38
    tag = "ADDRESS"

    bio = span_to_tag(BIO_SCHEME, text, [start], [end], [tag])

    print(bio)

    expected = ['O', 'O', 'O', 'B-ADDRESS', 'I-ADDRESS', 'I-ADDRESS',
                'I-ADDRESS', 'I-ADDRESS', 'I-ADDRESS', 'O', 'O', 'O', 'O', 'O']
    assert bio == expected


def test_span_to_bio_single_at_end():
    text = "My name is Josh"
    start = 11
    end = 15
    tag = "NAME"

    bilou = span_to_tag(BIO_SCHEME, text, [start], [end], [tag], )

    print(bilou)

    expected = ['O', 'O', 'O', 'I-NAME']
    assert bilou == expected


def test_span_to_bilou_multiple_tokens():
    text = "My Address is 409 Bob st. Manhattan NY. I just moved in"
    start = 14
    end = 38
    tag = "ADDRESS"

    bilou = span_to_tag(BILOU_SCHEME, text, [start], [end], [tag])

    print(bilou)

    expected = ['O', 'O', 'O', 'B-ADDRESS', 'I-ADDRESS', 'I-ADDRESS',
                'I-ADDRESS', 'I-ADDRESS', 'L-ADDRESS', 'O', 'O', 'O', 'O', 'O']
    assert bilou == expected


def test_span_to_bilou_adjacent_entities():
    text = "Mr. Tree"
    start1 = 0
    end1 = 2
    start2 = 4
    end2 = 8

    start = [start1, start2]
    end = [end1, end2]

    tag = ["TITLE", "NAME"]

    bilou = span_to_tag(BILOU_SCHEME, text, start, end, tag)

    print(bilou)

    expected = ['U-TITLE', 'U-NAME']
    assert bilou == expected


def test_span_to_bilou_single_at_end():
    text = "My name is Josh"
    start = 11
    end = 15
    tag = "NAME"

    bilou = span_to_tag(BILOU_SCHEME, text, [start], [end], [tag])

    print(bilou)

    expected = ['O', 'O', 'O', 'U-NAME']
    assert bilou == expected


def test_span_to_bilou_multiple_entities():
    text = "My name is Josh or David"
    start1 = 11
    end1 = 15
    start2 = 19
    end2 = 26

    start = [start1, start2]
    end = [end1, end2]

    tag = ["NAME", "NAME"]

    bilou = span_to_tag(BILOU_SCHEME, text, start, end, tag)

    print(bilou)

    expected = ['O', 'O', 'O', 'U-NAME', 'O', 'U-NAME']
    assert bilou == expected


def test_span_to_bio_multiple_entities():
    text = "My name is Josh or David"
    start1 = 11
    end1 = 15
    start2 = 19
    end2 = 26

    start = [start1, start2]
    end = [end1, end2]

    tag = ["NAME", "NAME"]

    bilou = span_to_tag(scheme=BIO_SCHEME, text=text, start=start,
                        end=end, tag=tag)

    print(bilou)

    expected = ['O', 'O', 'O', 'I-NAME', 'O', 'I-NAME']
    assert bilou == expected


def test_span_to_bio_specific_input():
    text = "Someone stole my credit card. The number is 5277716201469117 and " \
           "the my name is Mary Anguiano"
    start = 80
    end = 93
    expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O', 'O', 'B-PERSON', 'I-PERSON']
    tag = ["PERSON"]
    bilou = span_to_tag(BIO_SCHEME, text, [start], [end], tag)
    assert bilou == expected


def test_span_to_bilou_specific_input():
    text = "Someone stole my credit card. The number is 5277716201469117 and " \
           "the my name is Mary Anguiano"
    start = 80
    end = 93
    expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O', 'O', 'B-PERSON', 'L-PERSON']
    tag = ["PERSON"]
    bilou = span_to_tag(BILOU_SCHEME, text, [start], [end], tag)
    assert bilou == expected


def test_span_to_bilou_adjecent_identical_entities():
    text = "May I get access to Jessica Gump's account?"
    start = 20
    end = 32
    expected = ['O', 'O', 'O', 'O', 'O', 'B-PERSON', 'L-PERSON', 'O', 'O', 'O']
    tag = ["PERSON"]
    bilou = span_to_tag(BILOU_SCHEME, text, [start], [end], tag)
    assert bilou == expected


def test_overlapping_entities_first_ends_in_mid_second():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [22, 25]
    end = [37, 37]
    scores = [0.6, 0.6]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'US_PHONE_NUMBER',
                'US_PHONE_NUMBER', 'US_PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(BIO_SCHEME, text, start, end, tag, scores,
                     io_tags_only=True)
    assert io == expected


def test_overlapping_entities_second_embedded_in_first_with_lower_score():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [22, 25]
    end = [37, 33]
    scores = [0.6, 0.5]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'PHONE_NUMBER',
                'PHONE_NUMBER', 'PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(BIO_SCHEME, text, start, end, tag, scores,
                     io_tags_only=True)
    assert io == expected


def test_overlapping_entities_second_embedded_in_first_has_higher_score():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [23, 25]
    end = [37, 28]
    scores = [0.6, 0.7]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'US_PHONE_NUMBER',
                'PHONE_NUMBER', 'PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(BIO_SCHEME, text, start, end, tag, scores,
                     io_tags_only=True)
    assert io == expected


def test_overlapping_entities_pyramid():
    text = "My new phone number is 1 705 999 774 8720. Thanks, cya"
    start = [23, 25, 29]
    end = [41, 36, 32]
    scores = [0.6, 0.7, 0.8]
    tag = ["A1", "B2","C3"]
    expected = ['O', 'O', 'O', 'O', 'O', 'A1', 'B2', 'C3', 'B2',
                 'A1', 'O', 'O', 'O', 'O']
    io = span_to_tag(BIO_SCHEME, text, start, end, tag, scores,
                     io_tags_only=True)
    assert io == expected
