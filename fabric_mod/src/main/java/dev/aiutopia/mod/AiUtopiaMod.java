package dev.aiutopia.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

public class AiUtopiaMod implements ModInitializer {
    public static final String MOD_ID = "aiutopia";
    public static final Logger LOG = LoggerFactory.getLogger(MOD_ID);

    private GatewayServer gateway;
    private Py4JEntryPoint entryPoint;

    @Override
    public void onInitialize() {
        LOG.info("AI Utopia mod initializing");

        int py4jPort = Integer.parseInt(
            System.getProperty("aiutopia.py4j.port", "25001")
        );

        this.entryPoint = new Py4JEntryPoint();
        this.gateway    = new GatewayServer.GatewayServerBuilder(entryPoint)
                            .javaPort(py4jPort)
                            .build();

        ServerLifecycleEvents.SERVER_STARTING.register(server -> {
            entryPoint.attachServer(server);
            gateway.start();
            LOG.info("AI Utopia Py4J gateway listening on port {}", py4jPort);
        });

        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            try {
                gateway.shutdown();
            } finally {
                entryPoint.detachServer();
            }
            LOG.info("AI Utopia Py4J gateway stopped");
        });
    }
}
