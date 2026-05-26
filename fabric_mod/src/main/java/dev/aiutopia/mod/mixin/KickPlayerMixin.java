package dev.aiutopia.mod.mixin;

import com.mojang.authlib.GameProfile;
import dev.aiutopia.mod.agent.AgentRegistry;
import net.minecraft.server.command.KickCommand;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.Collection;

/** Block `/kick` of registered AI agents — for ALL permission levels.
 *
 *  Permission semantics: vanilla `/kick` already requires permission
 *  level 3, so any source that reaches this mixin is already op-3+.
 *  Filtering by permission level here would be a no-op. Instead we
 *  block ALL `/kick` of agents (including from full operators) and
 *  force the explicit `aiutopia agent kill <uuid>` CLI path, so
 *  accidental misclick kicks can never trigger permadeath.
 *
 *  Target verified in Step 2 against MC 1.21.1 Yarn mappings:
 *  KickCommand.kick(ServerCommandSource, Collection<GameProfile>, Text).
 */
@Mixin(KickCommand.class)
public abstract class KickPlayerMixin {

    @Inject(method = "kick", at = @At("HEAD"), cancellable = true)
    private static void aiutopia$blockKickOfAgents(
            ServerCommandSource source,
            Collection<GameProfile> targets,
            Text reason,
            CallbackInfoReturnable<Integer> cir) {
        for (GameProfile profile : targets) {
            if (AgentRegistry.isAgent(profile.getName())) {
                source.sendError(Text.literal(
                    "AI Utopia: cannot kick agent '" + profile.getName()
                    + "' via /kick. Use `aiutopia agent kill <uuid>` instead."));
                cir.setReturnValue(0);
                return;
            }
        }
    }
}
