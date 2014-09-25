function [iters,seeds] = solve1()
image = imread('scenery.jpg');
partSize = 256;
rows = floor(size(image,1)/partSize);
cols = floor(size(image,2)/partSize);
%size(image)
%rows
%cols
numOfParts = rows*cols;
%imshow(image);
%imwrite(image,'1.png','png');
pcs = cutImageToParts();
correctOrder = 1:numOfParts;
score = score_mgc();
%score = normalizeScores(score);
% 1 - left
% 2 - right
% 3 - top
% 4 - bottom
% score(i,j,1) = cost of placing part j to the left of part i
%%
score = score.^2;
%%
L = score(:,:,1);
R = score(:,:,2);
U = score(:,:,3);
D = score(:,:,4);
fprintf('Done Calculating Scores\n');

mutationRate = 0.1;
tournamentSize = 5;
popSize = 400;
mx = popSize/5;


%figure
num_runs = 30;
iters = zeros(1,num_runs);
seeds = [];

for runs = 1:num_runs
	fprintf('Run %d\n',runs);
	pop = zeros(popSize,numOfParts);
	popFitness = zeros(popSize,1);
	cumPopFitness = zeros(popSize,1);
	for i = 1:popSize, pop(i,:) = randperm(numOfParts); end

	num_iter = 500;
	frames = zeros(num_iter,numOfParts);
	
		% 100 iter good for partSize 
		for iter = 1:num_iter
			%fprintf('Iter %d\n',iter);
			evolve();
			frames(iter,:) = pop(1,:);
			%figure

			fg = disp(pop(1,:));
			if iter == 1,
				seeds = [seeds;pop(1,:)];
				imwrite(fg,strcat('scenery_',num2str(runs),'.png'),'png');
			end
			%calculateFitness(pop(1,:))
			%max(popFitness) - min(popFitness)
			%pause(0.01)
		
			if (pop(1,:) == correctOrder) break; end
		end
	iter
	iters(runs) = iter;
	save('frames2','frames');
	%pop(1,:)
	%figure
	%disp(pop(1,:));
	%calculateFitness(correctOrder)
	%calculateFitness(pop(1,:))
end

function fitness = calculateFitness(elem)
	elem_ = reshape(elem',cols,rows)';
	cost = 0;
	u = elem_(:,1:cols-1); v = elem_(:,2:cols);
	u = u(:); v = v(:);
	rCost = sum(R(sub2ind([numOfParts numOfParts],u,v)));
	lCost = sum(L(sub2ind([numOfParts numOfParts],v,u)));
	u = elem_(1:rows-1,:); v = elem_(2:rows,:);
	u = u(:); v = v(:);
	uCost = sum(U(sub2ind([numOfParts numOfParts],v,u)));
	dCost = sum(D(sub2ind([numOfParts numOfParts],u,v)));
	fitness = rCost + lCost + uCost + dCost;
end
function fitness = findFitness(elems)
	fitness = zeros(size(elems,1),1);
	for ct4 = 1:size(elems,1)
		fitness(ct4) = calculateFitness(elems(ct4,:));
	end
end
function findPopFitness()
	for t = 1:popSize
		popFitness(t) = calculateFitness(pop(t,:));
	end
end
function evolve()
	findPopFitness();
	%cumPopFitness = cumsum(popFitness/sum(popFitness));
	newpop = zeros(size(pop));
	%max_ind = find(popFitness == min(popFitness));
	%newpop(1,:) = pop(max_ind(1),:);
	[~,max_ind] = sort(popFitness);
	max_ind = max_ind(1:mx);
	newpop(1:mx,:) = pop(max_ind,:);

	%for ct1 = 2:2:popSize
	%	parent1 = tournamentSelection();
	%	parent2 = tournamentSelection();
	%	%parent1 = fitPropSel();
	%	%parent2 = fitPropSel();
	%	[c1,c2] = crossover1(parent1,parent2);
	%	newpop(ct1,:) = mutate(c1);
    %    if(ct1 ~= popSize) newpop(ct1+1,:) = mutate(c2); end
    %end

	for ct1 = mx+1:popSize
		success = false;
		while ~success,
			parent1 = tournamentSelection();
			parent2 = tournamentSelection();
			%parent1 = fitPropSel();
			%parent2 = fitPropSel();
			if rand < 0.1, 	child = crossover1(parent1,parent2);
			else 			child = crossover(parent1,parent2);
			end
			child = crossover(parent1,parent2);
			elem = mutate(child);
			if ~sum(ismember(newpop(1:ct1-1,:),elem,'rows')) success = true; end
		end
		newpop(ct1,:) = elem;
	end

	%for ct1 = mx+1:popSize
	%	parent1 = tournamentSelection();
	%	parent2 = tournamentSelection();
	%	%parent1 = fitPropSel();
	%	%parent2 = fitPropSel();
	%	child = crossover(parent1,parent2);
	%	newpop(ct1,:) = mutate(child);
	%end


	pop = newpop;
end
function child = crossover2(p1,p2)
	a = 1;
	b = randi([1 numOfParts]);
	if a <= b, 	startPos = a; endPos = b;
	else 		startPos = b; endPos = a; end		
	child = zeros(1,numOfParts);
	child(startPos:endPos) = p1(startPos:endPos);
	remainingPos = setdiff(1:numOfParts,startPos:endPos);
	%child(remainingPos) = setdiff(p2,p1(startPos:endPos),'stable');
	child(remainingPos) = p2(~ismember(p2,p1(startPos:endPos)));
end
function child = crossover(p1,p2)
	a = randi([1 numOfParts]);
	b = randi([1 numOfParts]);
	if a <= b, 	startPos = a; endPos = b;
	else 		startPos = b; endPos = a; end		
	child = zeros(1,numOfParts);
	child(startPos:endPos) = p1(startPos:endPos);
	remainingPos = setdiff(1:numOfParts,startPos:endPos);
	%child(remainingPos) = setdiff(p2,p1(startPos:endPos),'stable');
	child(remainingPos) = p2(~ismember(p2,p1(startPos:endPos)));
end
function [c1,c2] = crossover1(p1,p2)
	O = randi([1 numOfParts]);
	X = rem(O-1,rows)+1;
	Y = floor((O-1)/cols+1);
	x = randi([0 cols-X]);
	y = randi([0 rows-Y]);
	p1_ = reshape(p1',cols,rows)';
	p2_ = reshape(p2',cols,rows)';
    p1_r = p1_(Y:Y+y,X:X+x);
    p2_r = p2_(Y:Y+y,X:X+x);
	r1 = setdiff(p1_r(:),p2_r(:));
	r2 = setdiff(p2_r(:),p1_r(:));
	c1 = p1_;
	c2 = p2_;
	for i = 1:length(r1)
		c1(c1==r2(i)) = r1(i);
		c2(c2==r1(i)) = r2(i);
	end
    %[Y+y,rows,X+x,cols]
	c1(Y:Y+y,X:X+x) = p2_(Y:Y+y,X:X+x);
	c2(Y:Y+y,X:X+x) = p1_(Y:Y+y,X:X+x);
    c1 = c1(:)';
    c2 = c2(:)';
end
function sel = tournamentSelection()
	randElemPos = randi(popSize,tournamentSize,1);
	randElemFit = popFitness(randElemPos,:);
	max_fit = find(randElemFit == min(randElemFit));
	sel = pop(randElemPos(max_fit(1)),:);	
end
function sel = fitPropSel()
	k = find(cumPopFitness>rand);
	sel = pop(k(1),:);
end
function mutated = mutate(elem)
	mutated = elem;
	for pos1 = 1:numOfParts
		if rand < mutationRate
			pos2 = randi([1 numOfParts]);
			temp = mutated(pos2);
			mutated(pos2) = mutated(pos1);
			mutated(pos1) = temp;
		end
	end
end
function fg = disp(order)
        fg = [];
        order(1:cols);
        for i = 1:rows
          fg = cat(1,fg,cat(2,pcs{order((1:cols)+(i-1)*cols)}));
        end
        imshow(fg);
        %imwrite(fg,'2.png','png');
end
function score = score_mgc()
	oppSide = [2 1 4 3];
	P = pcs;
	dummyDiffs = [ 0 0 0 ; 1 1 1; -1 -1 -1; 0 0 1; 0 1 0; 1 0 0 ; -1 0 0 ; 0 -1 0; 0 0 -1];
	A = cell(4,1); 
	for edge = [1 2 3 4] 
		A{edge}.pix = zeros(numel(P),partSize*3);           
		A{edge}.D_mu  = zeros(numel(P),3); 
		A{edge}.D_cov = zeros(3,3,numel(P));
	end
	for w = 1:1:numel(P)
		P1 = P{w};
		P1 = single(P1);
		for edge = [1 2 3 4]
			P1Dif = getPlane(P1,edge,1) - getPlane(P1,edge,2);
			P1D_mu =  mean(P1Dif);
			P1D_cov = cov(double([P1Dif;dummyDiffs]));
			
			pixels = getPlane(P1,edge,1);
			A{edge}.pix(w,:) = pixels(:);
			A{edge}.D_mu(w,:) = P1D_mu;
			A{edge}.D_cov(:,:,w) = P1D_cov;
		end
	end

	N =numel(P);
	score = zeros(N,N,4,'single'); 
	for ii = 1:1:N, P{ii} = single(P{ii}); end
	onemat = ones(size(P{1},1),1);   
	psize = size(P{1},1);

	for jj = 1:1:4
	  for ii = 1:1:N-1
	  	P1D_mu =    A{jj}.D_mu(ii,:);
	  	P1D_cov =   A{jj}.D_cov(:,:,ii);
	  	p1S =       A{jj}.pix(ii,:);
	  	score(ii,ii,:) = inf;             

	  	for kk = ii+1:1:N
	  		s = oppSide(jj);

	  		P2D_mu =    A{s}.D_mu(kk,:);
	  		P2D_cov =   A{s}.D_cov(:,:,kk);
	  		p2S =       A{s}.pix(kk,:);

	  		% now, compute the score:
	  		P12DIF = p1S-p2S;
	  		P12DIF = reshape(P12DIF,partSize,3);
	  		P21DIF = -P12DIF;
	  		
	  		D12 = (P12DIF-(onemat*P2D_mu))/(P2D_cov); D12 = sum(D12 .* (P12DIF-(onemat*P2D_mu)),2);
	  		D21 = (P21DIF-(onemat*P1D_mu))/(P1D_cov); D21 = sum(D21 .* (P21DIF-(onemat*P1D_mu)),2);

	  		Dist = sum(sqrt(D12)+sqrt(D21));           

	  		score(ii,kk,jj)           = single(Dist);
	  		score(kk,ii,oppSide(jj))  = single(Dist);
	  	end
	  end
	end
	score(N,N,:) = inf;

	function plane = getPlane(P1,edge,n)
		if      (edge==3), plane = squeeze(P1(n,:,:));
		elseif  (edge==2), plane = squeeze(P1(:,end+1-n,:));
		elseif  (edge==4), plane = squeeze(P1(end+1-n,:,:));
		elseif  (edge==1), plane = squeeze(P1(:,n,:));
		end
	end
end
function [partsCompVal] = normalizeScores(partsCompVal)
  t = 0.000000000000001; 
  SCO = partsCompVal;
  normSCO = SCO;
  for ii = 1:1:size(SCO,3) %over each possible arrangement.
      % fprintf('Processing Scores Matrix %d\n',ii);
      %    ii
      [aaa,bbb] = sort(SCO(:,:,ii),2); % sorted over each row.
      rowmins = aaa(:,1:2); %2smallest over each row.
      rowminloc = bbb(:,1); %location of minimum in each row.
      
      [aaa,bbb] = sort(SCO(:,:,ii)); % sorted over each column.
      colmins = aaa(1:2,:); %2smallest over each column.
      colminloc = bbb(1,:); %location of minimum in each column.
      
      for jj = 1:1:size(SCO,1) %over each row.
          values = SCO(jj,:,ii);% the values in the row.
          rowmins(jj,1);
          n1 = values.*0 + rowmins(jj,1); %the minimum for that row...
          n1(rowminloc) = rowmins(jj,2); % the second lowest
          %each position can also be replaced by the smallest nonsame value in  the column...
          n2 = values.*0 + colmins(1,:); %the smallest value in each column.
          % but whereever the row is the same, we use the second lowest value instead
          n2(jj==colminloc)= colmins(2,jj==colminloc);
          nval = (values+t)./(min([n1;n2])+t);
          normSCO(jj,:,ii) = nval;
      end
  end
  partsCompVal = normSCO;
end
function pcs = cutImageToParts()
	for index = 1 : numOfParts
	  rowStartIndex = (ceil(index / cols)  - 1) * partSize + 1;
	  rowEndIndex = rowStartIndex + (partSize -  1);
	  colStartIndex = mod(index - 1, cols)  * partSize + 1;
	  colEndIndex = colStartIndex + (partSize -  1);
	  pcs{index} = image(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :);
	end
end

end
