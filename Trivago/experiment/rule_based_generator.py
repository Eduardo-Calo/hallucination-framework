def generate_accommodation_description_v2(accommodation_dict):
    """
    Takes a dict as input and returns a textual description
    :param accommodation_dict: dict
    :return: str
    """

    for key, value in accommodation_dict.items():
        if isinstance(value, list):
            accommodation_dict[key] = [v.lower() for v in value]

    # create description
    description = ''

    for category, values in accommodation_dict.items():
        # basic info
        if category == 'name':
            description += 'The ' + values + ' is a ' + str(int(accommodation_dict['star_rating'])) + ' star ' + accommodation_dict['accommodation_type'].lower() + ' located in ' + accommodation_dict['city'] + ', ' + accommodation_dict['country'] + '. '
        elif category == 'hotel_facilities':
            if len(values) > 0:
                description += 'Hotel facilities include ' + ', '.join(values) + '. '
        elif category == 'room_amenities':
            if len(values) > 0:
                description += 'Room amenities include ' + ', '.join(values) + '. '
        elif category == 'sport':
            if len(values) > 0:
                description += 'Sports facilities include ' + ', '.join(values) + '. '
        elif category == 'childcare_services':
            if len(values) > 0:
                description += 'Childcare services include ' + ', '.join(values) + '. '
        elif category == 'wellness':
            if len(values) > 0:
                description += 'Wellness facilities include ' + ', '.join(values) + '. '
        elif category == 'accessibility':
            if len(values) > 0:
                description += 'Accessibility features include ' + ', '.join(values) + '. '

    return description


# Test the function with the provided example input
example_input_v2 = {
    'name': 'Aktiv Panoramahotel Daniel',
    'star_rating': 4,
    'city': 'Sautens',
    'country': 'Austria',
    'accommodation_type': 'Hotel',
    'hotel_facilities': ['Hotel safe'],
    'room_amenities': ['Fridge', 'Cable TV'],
    'sport': ['Volleyball', 'Pool table'],
    'childcare_services': ['Childcare'],
    'wellness': ['Beauty salon', 'Steam room', 'Body treatments'],
    'accessibility': ['Wheelchair accessible']
}

print(generate_accommodation_description_v2(example_input_v2))