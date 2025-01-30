def secondary_time(type, alias):
    html = (
        "<View className=\"cause_effect_time\">",
        f"<Choices name='{alias}-time' toName=\"text\" showInLine=\"true\" visibleWhen=\"choice-selected\" whenTagName=\"{type}\" whenChoiceValue=\"{alias}\">",
        "<Choice alias='past' value='Past' />",
        "<Choice alias='present' value='Present' />",
        "<Choice alias='future' value='Future' />",
        "<Choice alias='general' value='N/A' />",
        "</Choices>",
        "</View>"
    )
    return "\n".join(html)


def make_choice(type, data):
    alias = data[0]
    data = data[1:]
    choice_name = data[0][:data[0].index(":")].strip()    
    data[0] = data[0][data[0].index(":")+1:].strip()

    help_text = []
    for i, text in enumerate(data[1:]):
        help = f"<Header size='5' value=\"{text}\"/>"
        help_text.append(help)
    help_text = "\n".join(help_text)

    html = (
        "<View>",
        f"<Choice hotkey=\"\" alias=\"{alias}\" value=\"{choice_name}\"/>\n",
        secondary_time(type, alias),
        "<Collapse bordered='false'>",
        f"<Panel value=\"{data[0]}\">",
        "<View>",
        help_text,
        "</View>",
        "</Panel>",
        "</Collapse>",
        "</View>",
    )
    
    return "\n".join(html)

def make_cause_effect_interface(type, data):
    choices=[]
    for i, d in enumerate(data):
        result = make_choice(type, d)
        choices.append(result)
    choices_html = "\n".join(choices)
    html = (
        "<View style=\"padding: 0 0 0 20px; margin-top: 0em; border-radius: 5px;\">",
        f"<Choices name=\"{type}\" toName=\"text\" choice=\"multiple\" showInLine=\"false\" visibleWhen=\"choice-selected\" whenTagName=\"narrative-type\" whenChoiceValue=\"{type.capitalize()}\">",
        f"<Header value=\"{type.capitalize()+'s'} of Inflation (select all that apply)\" />",
        choices_html,
        "</Choices>",
        "</View>"
    )
    return "\n".join(html)

def make_interface(causes, effects):
    html = (
        "<View>",
        make_cause_effect_interface('cause', causes),
        make_cause_effect_interface('effect', effects),
        "</View>"
    )

    html = "\n".join(html)
    return html

if __name__ == "__main__":
    causes = []
    with open("labels/cause_labels.txt", 'r') as f:
        cause = []
        for line in f:
            if line.strip() == "":
                causes.append(cause)
                cause=[]
            else:
                cause.append(line.strip())

    effects = []
    with open("labels/effect_labels.txt", 'r') as f:
        effect = []
        for line in f:
            if line.strip() == "":
                effects.append(effect)
                effect=[]
            else:
                effect.append(line.strip())

    final_html = make_interface(causes, effects)

    with open("generated_interface.html", 'w') as f:
        f.write(final_html)


# inp = \
# """
# Other Effects: Effects not included in above.
# Could include:
# Stale menu prices - If changing prices (e.g., on menus) is costly, inflation leads to inaccurate listed prices. Once stale prices become too costly for a business to maintain, it will be forced to more frequently update its prices.
# Rush to borrow - Possible for consumers to try to borrow more while rates are still low before they rise with inflation
# Shoe leather costs - Elevated inflation may encourage households to visit banks or ATMs more frequently to withdraw more cash necessary to purchase goods at higher prices. This more frequent transportation can be costly. Less relevant if households rely less on cash and more on electronic payments.
# Rise of substitute currencies - high inflation environments may increase appeal of alternative currencies like foreign currencies or cryptocurrency
# """


# # Example usage:
# #input_strings = ["Hello", "World", "Python"]
# input_strings = inp.strip().split('\n')
# html_result = strings_to_html(input_strings)
# print(html_result)

