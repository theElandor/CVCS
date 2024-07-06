+ Train 1
Allenato con SGD per 10 epoche, momentum = 0.9 e learning rate 0.01. Apparentemente sembra
scendere, ma in maniera molto instabile.
-------------------------------------------------------------
+ Train 2
Allenato con Adam per 20 epoche
opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
sembra andare addirittura peggio di prima.
Nonostante stia trainando con il batch norm la discesa della loss è estremamente instabile,
bisogna trovare una soluzione.
-------------------------------------------------------------
+ Train 3
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)
epochs = 30
Plottando la loss media per epoca sembra scendere alla fine. Direi che si debba solo trainare per più epoche,
servono molte ore su questa macchina.
-------------------------------------------------------------
+ Train 4 (urnet1)
Nonostante la loss sia scesa a 0.4, quindi manca ancora un po', l'ouput è qualcosa di sensato. Per ora ci stiamo
limitando a Postdam, non abbiamo ancora considerato le immagini low-quality dell'ESA.
Devo provare a trainare per almeno 150 epoche, ci vuole tipo un giorno in locale. Serve accedere ai server di Baraldi al più presto.
-------------------------------------------------------------
+Train 5 (urnet2)
epochs = 150
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)
Il modello, un checkpoint e i valori della loss sono stati salvati in locale.
La loss è scesa fino a 0.079!
Alcuni esempi di testing si possono trovare nella cartella train5.
Il modello è stato validato: l'mIoU è pari a 51,3%(opzione 'macro' del jsc score.) Il valore è in linea con quello riportato nel paper di Bahr (ovviamente per l'Unet classica), 
lui ottiene un 53% ma traina per molto più tempo e con anche i dati di DeepGlobe.
Se invece si tiene conto del non bilanciamento delle label, la mIoU è pari al 78,3%(opzione 'weighted' del jsc score).
-------------------------------------------------------------
+Train 6 (urnet3)
Provo ad allenare lo stesso modello from scratch per lo stesso numero di epoche usando però il dataset aumentato:
sono quindi stati aggiunti 2400 esempi contenenti le stesse immagini ma con aggiunta di immagini con noise.
Considerando che il dataset è il doppio più grande, per vedere lo stesso numero di esempi il training deve durare 75 epoche.
Verrà comunque salvato un checkpoint in caso si voglia allenare ulteriormente il modello.
--> come prevedibile la loss non è scesa a zero: serve più tempo per imparare a segmentare gli esempi distorti.
mIoU bilanciata --> 77.3%
mIoU non bilanciata --> 48.9%
Train 7
Per provare a migliorare la mIoU provo a trainare per altre 75 epoche.
(ricorda di salvare le loss in un altro file altrimenti si sovrascrivono le vecchie)
--> il modello allenato per un totale di 150 epoche con dataset aumentato è stato salvato come urnet3.2
mIoU non bilanciata --> 66%
mIoU bilanciata --> 87.3%
l'accuracy è migliorata notevolmente. Da considerare il fatto che nel test set sono presenti sia esempi normali
che esempi con rumore (gaussian blur e elastic deformation). Per confrontare questa accuracy con quella della vecchia rete (urnet2)
sarebbe opportuno testarla soltanto su esempi non noisy. I risultati sono comunque incoraggianti.
Sarebbe doveroso valutare l'accuracy su (i) test dataset di base (ii) test dataset solo noisy (iii) test dataset misto per avere valori
precisi di riferimento
In effetti la precedente precision del 51,3% non è propriamente comparabile con la precision attuale del 66%, perchè il modello ha visto
esempi diversi in training e in testing. Non è possibile neanche validare il modello con lo stesso split di validation di quello precedente,
perchè alcuni di questi esempi potrebbero essere stati visti dal modello attuale in training. --> da mettere a posto e pensare ad una soluzione
-------------------------------------------------------------
