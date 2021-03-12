%% ELEC 4700 PA -8 CMPA 
%Saifuddin Mohammed, #101092039

set(0,'defaultaxesfontsize',20)
set(0,'defaultaxesfontname','Times New Roman')
set(0,'DefaultLineLineWidth', 2);

set(0,'DefaultFigureWindowStyle','docked')


%Part 1 - Utilizng the Diode Equation to generate some data

V = linspace(-1.95,0.7,200);   %V vector creation

%Given Paramters
I_s = 0.01e-12;
I_b = 0.1e-12;
V_b = 1.3; 
G_p1 = 0.1; 

I_D = (I_s*(exp(1.2*V/0.025)-1)) + (G_p1*V) - (I_b*(exp(-1.2*(V+V_b)/0.025)-1));
I = I_D;

rand_vector = (1.2-0.5).*rand(size(I)) + 0.5;

I_new = rand_vector.*I;
figure(1)
subplot(2,1,1)
plot(V,I,V,I_new)
subplot(2,1,2)
semilogy(V,abs(I),V,abs(I_new))



%Part 2- Polynomial Fitting 
P_4 = polyfit(V,I,4);
Poly_4 = polyval(P_4,V);


P_8 = polyfit(V,I,8);
Poly_8 = polyval(P_8,V);



figure(2)
subplot(2,2,1)
plot(V,I,'b',V,Poly_4,'r')

subplot(2,2,2)
plot(V,I,'b',V,Poly_8,'r')

pol4 = polyfit(V,I_new,4); 
a = polyval(pol4,V);
subplot(2,2,3)
semilogy(V,abs(I),'r',V,abs(a),'b')


pol8 = polyfit(V,I_new,8);
b = polyval(pol8,V);
subplot(2,2,4)
semilogy(V,abs(I),'b',V,abs(b),'r')




% Part 3- Non-Linear Polynomial Fitting 

% Fitting 1
fo_1 = fittype('A*(exp(1.2*x/0.025)-1) + (0.1*x) - (C*(exp(-1.2*(x+1.3)/0.025)-1))');
ff1 = fit(V',I',fo_1);
If1 = ff1(V);

figure(3)
subplot(3,1,1)
semilogy(V,abs(If1'),'r',V,abs(I_new),'b')


%Fitting 2
fo_2 = fittype('A*(exp(1.2*x/0.025)-1) + (B*x) - (C*(exp(-1.2*(x+1.3)/0.025)-1))');
ff_2 = fit(V',I',fo_2);
If_2 = ff_2(V);

subplot(3,1,2)
semilogy(V,abs(If_2'),'r',V,abs(I_new),'b')

%Fitting 3
fo_3 = fittype('A*(exp(1.2*x/0.025)-1) + (B*x) - (C*(exp(-1.2*(x+D)/0.025)-1))');
ff_3 = fit(V',I',fo_3);
If_3 = ff_3(V);

subplot(3,1,3)
semilogy(V,abs(If_3'),'b',V,abs(I_new),'r')




% Part 4 - Fitting Using Neural Net Model

inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize); 
net.divideParam.trainRatio = 70/100; 
net.divideParam.valRatio = 15/100; 
net.divideParam.testRatio = 15/100; 
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets); 
performance = perform(net,targets,outputs); 
view(net)
Inn = outputs;

figure(4)
plot(inputs,Inn,'b',inputs,I_new,'r')
