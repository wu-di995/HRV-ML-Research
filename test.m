path = "C:\Users\Wu Di\Documents\HRV-ML-Research\RR_edited\s01-Headarray - Command_edit-RR.csv";
csv = readtable(path);
time = table2array(csv(:,2));
RR = table2array(csv(:,3));