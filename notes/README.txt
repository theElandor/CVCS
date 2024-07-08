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
+Train 6, Train 7 prove con dataset aumentato (urnet3.1, 3.2), non rilevanti.
-------------------------------------------------------------
+Train 8 (urnet3.3)
Usando il dataset aumentato durante il training:
+ Validation su dataset con immagini originali (mIoU): 49% (circa in linea con quello precedente). Mi aspettavo meglio in relatà.
  weigthed IoU --> 76%
+ Validation su dataset con immagini distorte (mIoU): 44%. Abbastanza buono questo, probabilmente il modello 'sacrifica' un po' di precisione
  sulle immagini non distorte per guadagnarci in generalità, in modo da segmentare bene quelle distorte.
  weigthed IoU --> 72%
Per quanto riguarda la gestione del dataset e dello split train / validation:
sono stati creati due datasets: il primo con le 2400 immagini originali e relative maschere, il secondo con le 2400 immagini distorte e relative maschere.
I due dataset sono stati concatenati, ottenendo un dataset con 4800 elementi: gli indici da [0, 2400) corrispondono alle imm. originali, da [2400, 4800) alle immagini distorte.
Sono stati creati poi gli split di train e validation:
1) Train: i primi 2400 indici sono stati shufflati usando uno specifico seed, e sono stati presi i primi 1920. Sono poi stati aggiunti tutti gli indici 
   delle immagini distorte corrispondenti, semplicemente aggiungendo un'offset di 2400 a ogni indice originale.
   E' stato creato un SubsetRandomSampler che durante l'allenamento sceglie casualmente batch_size indici ad ogni iterazione di train.
2) Validation base: sono stati presi i 480 indici rimanenti dalla lista di immagini originali
3) Validation noisy: indici del validation base + 2400, ossia i rimanenti che non erano ancora stati scelti tra gli indici delle immagini distorte.
