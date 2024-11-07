package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import org.example.processoremulatorfx.App;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.Command;
import org.example.processoremulatorfx.IObserver;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

import java.io.IOException;

public class MainViewController implements IObserver {
  ProgramModel programModel = BProgramModel.build();

  @FXML
  void initialize()
  {
    programModel.addObserver(this);

    try {
      AddCommandViewController pc1 = new AddCommandViewController();
      FXMLLoader fxmlLoader1 = new FXMLLoader(
              App.class.getResource("Views/AddCommandView.fxml"));

      ConsoleViewController pc2 = new ConsoleViewController();
      FXMLLoader fxmlLoader2 = new FXMLLoader(
              App.class.getResource("Views/ConsoleView.fxml"));

      MemoryViewController pc3 = new MemoryViewController();
      FXMLLoader fxmlLoader3 = new FXMLLoader(
              App.class.getResource("Views/MemoryView.fxml"));

      ProcessorEmulatorViewController pc4 = new ProcessorEmulatorViewController();
      FXMLLoader fxmlLoader4 = new FXMLLoader(
              App.class.getResource("Views/ProcessorEmulatorView.fxml"));

      RegistersViewController pc5 = new RegistersViewController();
      FXMLLoader fxmlLoader5 = new FXMLLoader(
              App.class.getResource("Views/RegistersView.fxml"));

      StatisticsViewController pc6 = new StatisticsViewController();
      FXMLLoader fxmlLoader6 = new FXMLLoader(
              App.class.getResource("Views/StatisticsView.fxml"));

      FXMLLoader fxmlLoader7 = new FXMLLoader(
              App.class.getResource("Views/InfoView.fxml"));

      FXMLLoader fxmlLoader8 = new FXMLLoader(
              App.class.getResource("Views/ProgramStatusView.fxml"));

      Pane pane8 = fxmlLoader8.load();
      Pane pane1 = fxmlLoader1.load();
      Pane pane2 = fxmlLoader2.load();
      Pane pane3 = fxmlLoader3.load();
      Pane pane4 = fxmlLoader4.load();
      Pane pane5 = fxmlLoader5.load();
      Pane pane6 = fxmlLoader6.load();
      //Pane pane7 = fxmlLoader7.load();

      VBox vBox = new VBox(2);
      vBox.setSpacing(0);
      vBox.getChildren().addAll(pane4, pane5);

      main.add(pane1, 1, 0);
      main.add(pane2, 1, 2);
      main.add(pane3, 0, 1);
      main.add(vBox, 0, 0);
      main.add(pane6, 1, 1);
      //main.add(pane7, 0, 2);
      main.add(pane8, 0, 2);

    } catch (Exception e) {
        throw new RuntimeException(e);
    }
  }

  @FXML
  GridPane main;

  @FXML
  GridPane allCommands;

  @Override
  public void event(ProgramModel pm) {
    allCommands.getChildren().clear();
    for (Command c: pm) {
      CommandViewController pc = new CommandViewController();
      FXMLLoader fxmlLoader = new FXMLLoader(
              App.class.getResource("Views/CommandView.fxml"));
      fxmlLoader.setController(pc);
      try {
        Pane pane = fxmlLoader.load();
        pc.setCommand(c);
        allCommands.addColumn(0, pane);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  @Override
  public void event(CpuModel c) {}

  @Override
  public void event(ExecuterModel e) {}
}
