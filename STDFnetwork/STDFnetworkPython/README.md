STDFnetwork implemented in Python 

COMENTARI:
Teníem dubtes perquè jo havia deixat un espai en blanc que separa neurones en el vector de la xarxa x0, així iteram de 20 equacions en 20 equacions per recórrer les neurones, però només contàvem 19 variables. El teu codi crec que també té neq igual a 20, maybe és perquè vares dir que podem canviar la Prel per una altra equació que s'hauria d'integrar i ja vares deixar l'espai en blanc tenint això present?? Així si hem de canviar l'equació que descriu Prel, l'afegim a l'espai en blanc?? No sé ehh, per si té sentit.

@CatiVich hem de comprovar aquestes tres coses, NO ME QUADREN (en teoria està com el teu codi):
1. NetworkSTDall.py line 99: Els nombres random del voltatge inicial se surten dels límits que diu al paper, no és greu    però per fer-ho bé...

2. NetworkField.py  line 103 + bucle for petit a continuacio: 

    2.1. El bucle aquest només mira sAMPA, sNMDA, sGABA, per calcular els 'fact', què passa amb les inhibitory synAMPA, synNMDA, synGABA ?? Aquestes variables s'instancien al principi però en aquest bucle no s'empren (s'empren per definir les equacions de les inhibitory ok). Però, per què al bucle for que comença a la line 103, per calcular els 'fact' només es miren les sAMPA, sNMDA, sGABA que són de les excitatory??? Al teu codi tens dos fors, un per excitatory i s'altre per inhibitory, però en ambdós mires només aquestes variables que he dit a dalt.
    
    2.2. Dins el mateix for que comença a la line 103, quan indexa la matriu P, el meu cap ho pensa alrevés, fer P[presyn_neuron,postsyn_neuron] perquè les files representen les presynaptiques (que se recorren per calcular els fact) i les columnes les postsinaptiques, però no codi va alrevés, no understand (o ho estic pensant alrevés).

3. Prerelease.py line 34, només que a facilitació el paper posa que p0=0.1 i nosaltres tenim p0=1 tant en les variables de facilitació com en el de depressió.