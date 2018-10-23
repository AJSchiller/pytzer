cd 'E:\Dropbox\_UEA_MPH\pytzer'

% Select test and reference electrolytes
tst = 'KCl';
ref = 'NaCl';

% Get electrolyte-specifics
tsttit = tst;
switch tst
    case {'KCl' 'NaCl'}
        fxl = [0 2.5];
        fxt = 0:0.5:2.5;
        fyl = 0.012 * [-1 1];
        fyt = -0.012:0.006:0.012;
        fyl2 = [0 0.008];
        fyt2 = 0:0.002:0.008;
    case 'CaCl2'
        tsttit = 'CaCl_2';
        fxl = [0 2];
        fxt = 0:0.5:2.5;
        fyl = 0.012 * [-1 1];
        fyt = -0.012:0.006:0.012;
        fyl2 = [0 0.008];
        fyt2 = 0:0.002:0.008;
end %switch

% Load data from simpar_iso.py
froot = 'pickles/simpar_iso_';
isobase = readtable([froot 'isobase_t' tst '_r' ref '.csv']);
load([froot 'pshape_t' tst '_r' ref '.mat'])
load([froot 'isoerr_t' tst '_r' ref '.mat'])

% Load data from simloop_iso.py
load(['pickles/simloop_iso_bC_t' tst '_r' ref '_10.mat'])

% Define marker styles
srcs = unique(isobase.src);
[fmrk,fclr,fmsm] = simpar_markers(srcs);
mksz = 10;

figure(7); clf
printsetup(gcf,[9 12])

subplot(2,2,1); hold on

xlim(fxl)
ylim(fyl)

patch([tot; flipud(tot)],[sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)    
    
% isobase = readtable('pickles/isobase_test.csv');
    
    for S = 1:numel(srcs)

        SL = strcmp(isobase.src,srcs{S});

        scatter(sqrt(isobase.(tst)(SL)),isobase.(['dosm_' tst])(SL), ...
            mksz*fmsm.(srcs{S}),fclr.(srcs{S}),'filled', ...
            'marker',fmrk.(srcs{S}), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)

%         scatter(sqrt(isobase.(tst)(SL)),isobase.dosm25_sim(SL), ...
%             mksz*fmsm.(srcs{S}),fclr.(srcs{S}),'filled', ...
%             'marker',fmrk.(srcs{S}), ...
%             'markerfacealpha',0.7, 'markeredgealpha',0)

        TL = tot >= min(isobase.(tst)(SL)) & tot <= max(isobase.(tst)(SL));
        plot(sqrt(tot(TL)),isoerr_sys.(srcs{S}) ...
            * (1 + 1./(tot(TL) + 0.03)), ...
            'color',fclr.(srcs{S}))

    end %for S
    
    plot(get(gca,'xlim'),[0 0],'k')
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',fyt)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.1f'))
    
    xlabel(['[\itm\rm(' tsttit ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    ylabel('\Delta\phi \times 10^3')

    % %% Components
    % plot(tot,dosm_dtot *5e-3, 'linewidth',2)
    % plot(tot,dosm_dtotR*1e-2, 'linewidth',2)
    % plot(tot,(dosm_dtot + dosm_dtotR)*5e-3, 'linewidth',2)
    % plot(tot,4e-4*(1 + 1./(tot+0.03)),'k', 'linewidth',2)
    
    text(0,1.09,'(a)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    spfig = gca;

subplot(2,2,3); hold on

    xlim(fxl)
    ylim(fyl2)

    for S = 1:numel(srcs)

        SL = strcmp(isobase.src,srcs{S});

        scatter(sqrt(isobase.(tst)(SL)), ...
            abs(isobase.(['dosm_' tst '_sys'])(SL)), ...
            mksz*fmsm.(srcs{S}),fclr.(srcs{S}),'filled', ...
            'marker',fmrk.(srcs{S}), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)

        TL = tot >= min(isobase.(tst)(SL)) & tot <= max(isobase.(tst)(SL));
        nl = plot(sqrt(tot(TL)),isoerr_rdm.(srcs{S})(1) ...
            + isoerr_rdm.(srcs{S})(2) ./ (tot(TL) + 0.03), ...
            'color',fclr.(srcs{S})); nolegend(nl)
        
    end %for S
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',fyt2)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.1f'))
    
    xlabel(['[\itm\rm(' tsttit ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    ylabel('|\Delta\phi - \delta_{ISO}XXX| \times 10^3')
    
    text(0,1.09,'(b)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    spfg2 = gca;

subplot(2,2,2); hold on    
    
    setaxes(gca,8)
    set(gca, 'xtick',[], 'ytick',[], 'box','on')
    
    for S = 1:numel(srcs)
        
        src = srcs{S};
        
        scatter(0.6,numel(srcs)-S, mksz*fmsm.(src)*1.5,fclr.(src), ...
            'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        text(1.1,numel(srcs)-S,src, 'fontname','arial', 'fontsize',8, ...
            'color','k')
        
    end %for S
    
    xlim([0 5])
    ylim([-0.75 numel(srcs)-0.25])
    
    spleg = gca;
    
% Positioning    
spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.08 0.6 0.35];
spleg.Position = [0.8 0.63 0.18 0.25];

% print('-r300',['figures/simpar_iso_t' tst '_r' ref],'-dpng')
