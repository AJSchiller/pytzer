%% Load Python outputs
load('pickles/simpar_fpd.mat');
fpdbase = readtable('pickles/simpar_fpd.csv');
fpdsrcs.all.srcs = unique(fpdbase.src);

%% Plot raw FPD data

% Choose electrolyte to plot
eles = {'KCl' 'NaCl'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(fpdsrcs.all.srcs)
    fmrk.(fpdsrcs.all.srcs{S}) = mrks{S};
    fclr.(fpdsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(fpdsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 10;

for E = 2%:numel(eles)
ele = eles{E};

% Define settings that depend upon electrolyte
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:5;
        fyl = 0.35*[-1 1];
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:6;
        fyl = 0.3*[-1 1.0000001];
end %switch

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(2); clf
printsetup(gcf,[9 6])
flegs = {};

subplot(1,2,1); hold on

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
%         SL = SL & fpdbase.t == 298.15;
        
        scatter(fpdbase.m(SL),fpdbase.dfpd(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
            Sy = fpderr_sys.(ele).(src)(2) .* Sx ...
                + fpderr_sys.(ele).(src)(1);
            nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
            flegs{end+1} = src;
        end %if
            
    end %for S
    
    xlim(fxl)
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.1:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\Delta\itd\rm / K')
    
    text(0,1.09,'(a)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    spfig = gca;

subplot(1,2,2); hold on
    
    setaxes(gca,8)
    set(gca, 'xtick',[], 'ytick',[], 'box','on')
    
    for S = 1:numel(flegs)
        
        src = flegs{S};
        
        scatter(0.6,numel(flegs)-S, mksz*fmsm.(src)*1.5,fclr.(src), ...
            'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        text(1.1,numel(flegs)-S,src, 'fontname','arial', 'fontsize',8, ...
            'color','k')
        
    end %for S
    
    xlim([0 5])
    ylim([-0.75 numel(flegs)-0.25])
    
    spleg = gca;

% Positioning    
spfig.Position = [0.15 0.18 0.6 0.71];
spleg.Position = [0.8 0.31 0.18 0.45];

print('-r300',['figures/simpar_fpd_' ele],'-dpng')

end %for E
