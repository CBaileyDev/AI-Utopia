/* AI Utopia — shared mock data + design helpers (plain global script) */
(function () {
  const roleColors = {
    gatherer: '#00E5CC',
    builder: '#B07CFF',
    farmer: '#66BB6A',
    defender: '#EF5350',
  };

  const roleMeta = {
    gatherer: { label: 'Gatherer', icon: 'axe' },
    builder: { label: 'Builder', icon: 'hammer' },
    farmer: { label: 'Farmer', icon: 'wheat' },
    defender: { label: 'Defender', icon: 'shield' },
  };

  const agents = [
    { id: '1', name: 'Oak', role: 'gatherer', status: 'alive', uuid: '01HABCDE1234567890ABCDEF', skin: 'Default Villager', born: '2026-05-26T14:23:00Z', x: 142, z: -89, rewards: 342, health: 20, hunger: 18 },
    { id: '2', name: 'Stone', role: 'builder', status: 'training', uuid: '01HABCDE1234567890ABCDF0', skin: 'Steve Variant', born: '2026-05-27T09:15:00Z', x: 89, z: 45, rewards: 198, health: 20, hunger: 16 },
    { id: '3', name: 'Wheat', role: 'farmer', status: 'alive', uuid: '01HABCDE1234567890ABCDF1', skin: 'Alex Variant', born: '2026-05-28T16:42:00Z', x: -34, z: 120, rewards: 156, health: 18, hunger: 20 },
    { id: '4', name: 'Iron', role: 'defender', status: 'idle', uuid: '01HABCDE1234567890ABCDF2', skin: 'Default Villager', born: '2026-05-29T11:08:00Z', x: 200, z: -150, rewards: 98, health: 20, hunger: 14 },
  ];

  const logs = [
    { id: '1', timestamp: '14:32:15', type: 'AGENT', message: 'Oak collected 12 oak_log at (142, 64, -89)' },
    { id: '2', timestamp: '14:32:10', type: 'TRAIN', message: 'Epoch 1247 complete — mean reward +2.14' },
    { id: '3', timestamp: '14:32:05', type: 'CHAT', message: "@Oak What are you working on? → 'Gathering wood for the village stockpile'" },
    { id: '4', timestamp: '14:31:58', type: 'SYSTEM', message: 'Py4J bridge heartbeat confirmed (12ms)' },
    { id: '5', timestamp: '14:31:45', type: 'AGENT', message: 'Stone placed 8 stone_bricks at village center (89, 64, 45)' },
    { id: '6', timestamp: '14:31:32', type: 'AGENT', message: 'Wheat harvested 6 wheat at farm plot (-34, 64, 120)' },
    { id: '7', timestamp: '14:31:20', type: 'TRAIN', message: 'Policy loss 0.0047 (down 12% from prev)' },
    { id: '8', timestamp: '14:31:15', type: 'SYSTEM', message: 'Memory compaction complete — 847 entries archived' },
    { id: '9', timestamp: '14:31:02', type: 'AGENT', message: 'Iron patrolling perimeter at (200, 70, -150)' },
    { id: '10', timestamp: '14:30:48', type: 'CHAT', message: "@Stone Need more stone → 'On it, mining at depth 32'" },
    { id: '11', timestamp: '14:30:31', type: 'TRAIN', message: 'KL divergence 0.0021 within trust region' },
    { id: '12', timestamp: '14:30:12', type: 'AGENT', message: 'Oak deposited 64 oak_log into village chest' },
  ];

  const epochs = Array.from({ length: 24 }, (_, i) => ({
    epoch: 1247 - i,
    meanReward: 2.14 + Math.sin(i * 0.3) * 0.4 - i * 0.01,
    policyLoss: 0.0047 + Math.cos(i * 0.5) * 0.001,
    valueLoss: 0.0089 + Math.sin(i * 0.4) * 0.002,
    entropy: 0.312 + Math.cos(i * 0.3) * 0.05,
    klDiv: 0.0021 + Math.sin(i * 0.6) * 0.0005,
    duration: (12 + Math.floor((Math.sin(i * 1.3) + 1) * 4)) + 's',
    status: (['improved', 'stable', 'degraded'])[Math.abs(Math.round(Math.sin(i * 2.1) * 1.4)) % 3],
  }));

  // Deterministic-ish activity timeline (avoid re-randomizing on every render)
  const activity = Array.from({ length: 14 }, (_, i) => {
    const hour = 14 - Math.floor((13 - i) / 2);
    const min = ((13 - i) % 2) * 30;
    const s = (n, p) => 4 + Math.round((Math.sin(i * 0.5 + p) + 1) * n);
    return {
      time: `${String(hour).padStart(2, '0')}:${min === 0 ? '00' : min}`,
      gatherer: s(7, 0), builder: s(5, 1.4), farmer: s(4, 2.6), defender: s(3, 3.9),
    };
  });

  const rewardCurve = Array.from({ length: 120 }, (_, i) => ({
    epoch: 1128 + i,
    reward: 1.45 + Math.sin(i * 0.09) * 0.28 + (i / 120) * 0.85 + Math.sin(i * 0.7) * 0.06,
    baseline: 1.8,
  }));

  window.AIU = {
    roleColors,
    roleMeta,
    getRoleColor: (r) => roleColors[r] || '#7A7A99',
    agents,
    logs,
    epochs,
    activity,
    rewardCurve,
    statusColor: {
      alive: '#00E5CC', training: '#FFB800', idle: '#7A7A99', offline: '#FF4466',
    },
  };
})();
