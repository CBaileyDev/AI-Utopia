package dev.aiutopia.mod.mixin;

import dev.aiutopia.mod.agent.AgentRegistry;
import dev.aiutopia.mod.chat.ChatEventBuffer;
import net.minecraft.network.message.SignedMessage;
import net.minecraft.server.network.ServerPlayNetworkHandler;
import net.minecraft.server.network.ServerPlayerEntity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** §3.3 — Intercept chat starting with `@<agent_name>`, emit ChatEvent to
 *  Python via ChatEventBuffer. Vanilla broadcast is suppressed when the
 *  pattern matches (suppressed_in_chat=true default in ChatEvent schema). */
@Mixin(ServerPlayNetworkHandler.class)
public abstract class ChatMessageMixin {
    @Shadow public ServerPlayerEntity player;

    private static final Pattern AGENT_MENTION =
        Pattern.compile("^@([A-Za-z0-9_]{1,16})\\s+(.+)$", Pattern.DOTALL);

    @Inject(method = "handleDecoratedMessage", at = @At("HEAD"), cancellable = true)
    private void aiutopia$interceptAgentMention(SignedMessage message, CallbackInfo ci) {
        String text = message.getSignedContent();
        if (text == null) return;

        Matcher m = AGENT_MENTION.matcher(text);
        if (!m.matches()) return;

        String agentName = m.group(1);
        String body      = m.group(2);
        if (!AgentRegistry.isAgent(agentName)) return;

        String json = String.format(
            "{\"sender_player_uuid\":\"%s\",\"sender_player_name\":\"%s\","
          + "\"addressed_agent_name\":\"%s\",\"text\":\"%s\","
          + "\"timestamp\":%d}",
          player.getUuidAsString(),
          escape(player.getGameProfile().getName()),
          escape(agentName),
          escape(body),
          System.currentTimeMillis() / 1000L
        );
        ChatEventBuffer.push(json);
        ci.cancel();    // suppress vanilla broadcast
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r");
    }
}
