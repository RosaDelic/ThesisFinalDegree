STDFnetwork implemented in Python 

@CatiVich hem de comprovar aquestes dues coses, NO ME QUADREN (en teoria està com el teu codi):
1. NetworkSTDall.py line 99: Els nombres random del voltatge inicial se surten dels límits que diu al paper, no és greu    però per fer-ho bé...

2. NetworkField.py  line 103: 

    2.1. El bucle aquest només mira sAMPA, sNMDA, sGABA, què passa amb les inhibitory synAMPA, synNMDA, synGABA ??
    
    2.2. Quan indexa la matriu P, el meu cap ho pensa alrevés, fer P[presyn_neuron,postsyn_neuron]
