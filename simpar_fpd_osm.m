%% Load Python outputs
pfpd = load('pickles/simpar_fpd_v2.mat');
pfpd = struct2table(pfpd.pshape_fpd);
fpdbase = readtable('pickles/simpar_fpd_v2.csv');
fpdsrcs.all.srcs = unique(fpdbase.src);

% Plot raw FPD data
figure(4); clf

subplot(2,1,1); hold on
    plot(pfpd.tot,pfpd.fpd_err_CaCl2-pfpd.fpd_CaCl2)
    L = strcmp(fpdbase.ele,'NaCl');
    scatter(fpdbase.m(L),fpdbase.dfpd(L))
    xlim([0 6.25])
    plot(get(gca,'xlim'),[0 0],'k')
    grid on

subplot(2,1,2); hold on
    plot(pfpd.tot,pfpd.osm25_fpd_NaCl-pfpd.osm25_calc_NaCl)
    plot(pfpd.tot,pfpd.osm_fpd_NaCl-pfpd.osm_calc_NaCl)
%     plot(pfpd.tot,pfpd.osm25_fpd_err_CaCl2-pfpd.osm25_calc_CaCl2)
    scatter(fpdbase.m(L),fpdbase.dosm25(L))
    xlim([0 6.25])
    ylim([-1 1]*1e-2)
    grid on

%% Choose electrolyte to plot
eles = {'KCl' 'NaCl' 'CaCl2'};

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

for E = 1:numel(eles)
ele = eles{E};

% Define settings that depend upon electrolyte
eletit = ele;
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:5;
        fyl = 0.05000000001*[-1 1];
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:6;
        fyl = 0.05000000001*[-1 1];
    case 'CaCl2'
        fxl = [0 7.5];
        fxt = 0:6;
        fyl = 0.05000000001*[-1 1];
        eletit = 'CaCl_2';
end %switch

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(E); clf
printsetup(gcf,[9 12])
flegs = {};

subplot(2,2,1); hold on

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
%         SL = SL & fpdbase.t == 298.15;
        
        scatter(fpdbase.m(SL),fpdbase.dosm25(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
%             SPL = pshape_fpd.tot >= min(fpdbase.m(SL)) ...
%                 & pshape_fpd.tot <= max(fpdbase.m(SL));
%             Sx = pshape_fpd.tot(SPL);
%             Sy = (fpderr_sys.(ele).(src)(2) .* Sx ...
%                 + fpderr_sys.(ele).(src)(1)) ...
%                 .* pshape_fpd.(['dosm25_' ele])(SPL);
% %             nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
% %                 'linewidth',0.5); nolegend(nl)
            flegs{end+1} = src;
        end %if
            
    end %for S
    
    xlim(fxl)
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.01:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.2f'))
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
    ylabel('\Delta\phi_{25}')
    
    text(0,1.09,'(a)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    spfig = gca;

subplot(2,2,2); hold on
    
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

subplot(2,2,3); hold on

    xlim(fxl)
    ylim([0.999999999e-5 1])

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',10.^(-6:0))
%     set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))
    set(gca, 'YScale','log')
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
    ylabel(['|\Delta\phi ' endash ' \itm\rm \delta_{FPD}| \times 10^{3}'])
    
    text(0,1.09,'(b)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
%     % Plot data by source
%     for S = 1:numel(fpdsrcs.(ele).srcs)
% 
%         src = fpdsrcs.(ele).srcs{S};
%         SL = EL & strcmp(fpdbase.src,src);
%         
%         scatter(fpdbase.m(SL),abs(fpdbase.dfpd_sys(SL)), ...
%             mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
%             'markeredgecolor',fclr.(src), ...
%             'markerfacealpha',0.7, 'markeredgealpha',0)
%         
%         if any(SL)
%             Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
%             Sy = fpderr_rdm.(ele).(src)(2) .* Sx ...
%                 + fpderr_rdm.(ele).(src)(1);
%             nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
%                 'linewidth',0.5); nolegend(nl)
%         end %if
%             
%     end %for S
    
    spfg2 = gca;
    
% Positioning    
spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.08 0.6 0.35];
spleg.Position = [0.8 0.63 0.18 0.25];

% print('-r300',['figures/simpar_fpd_osm_' ele],'-dpng')

end %for E
